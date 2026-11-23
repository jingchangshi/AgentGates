"""Tests for bounded build/source lock waiting in Engine.run."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_gates import engine as engine_mod
from agent_gates.engine import Engine, build_directory, environment
from agent_gates.evidence import EvidenceStore
from agent_gates.locking import ResourceLock
from agent_gates.registry import Registry
from agent_gates.manifest import digest


class BuildLockWaitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "input.txt").write_text("source bytes")
        self.pack = self.root / "adapters/test-pack"
        (self.pack / "scripts").mkdir(parents=True)
        self.script = self.pack / "scripts/gate.py"
        self.script.write_text(
            'import json,sys\nif "--check-config" in sys.argv: sys.exit(0)\nprint(json.dumps({"ok": True}))\n'
        )
        self.manifest = {
            "schemaVersion": 1,
            "id": "test-pack",
            "version": "1",
            "files": {},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [
                {
                    "id": "test.run",
                    "cwd": "source",
                    "timeout": 5,
                    "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
                    "configCheck": [sys.executable, "-I", "{pack}/scripts/gate.py", "--check-config"],
                }
            ],
            "gates": [
                {
                    "id": "test",
                    "recipeId": "test.run",
                    "gateType": "deterministic-test",
                    "observationChecks": {"ok": True},
                }
            ],
        }
        self._write_manifest()
        self.registry = Registry(self.root / "adapters")
        self.store = EvidenceStore(self.root / "evidence")
        self.engine = Engine(
            registry=self.registry,
            store=self.store,
            lock_root=self.root / "locks",
            device_lock_root=self.root / "locks",
        )

    def _write_manifest(self):
        self.manifest["files"] = {"scripts/gate.py": digest(self.script)}
        (self.pack / "manifest.json").write_text(json.dumps(self.manifest))

    def _build_resource(self) -> str:
        loaded = self.registry.get("test-pack")
        env = environment(loaded)
        return "build:" + str(build_directory(self.source, loaded, env))

    def test_bounded_lock_wait_succeeds_when_holder_releases_within_timeout(self):
        resource = self._build_resource()
        lock = ResourceLock(resource, self.root / "locks")
        lock.__enter__()
        released = threading.Event()

        def _release_soon():
            time.sleep(0.25)
            lock.__exit__(None, None, None)
            released.set()

        holder = threading.Thread(target=_release_soon, daemon=True)
        holder.start()
        try:
            run = self.engine.run("test-pack", self.source, ["test"])
        finally:
            holder.join(timeout=5)
            if not released.is_set():
                lock.__exit__(None, None, None)
        self.assertEqual(run["status"], "PASS", run)
        self.assertEqual(run["gates"][0]["observations"]["ok"], True)

    def test_bounded_lock_wait_fails_cleanly_after_timeout_expiry(self):
        resource = self._build_resource()
        short_engine = Engine(
            registry=self.registry,
            store=self.store,
            lock_root=self.root / "locks",
            device_lock_root=self.root / "locks",
            build_lock_timeout=0.15,
        )
        with ResourceLock(resource, self.root / "locks"):
            run = short_engine.run("test-pack", self.source, ["test"])
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertNotIn("process", run["gates"][0])
        failure = run["gates"][0]["failures"][0]
        self.assertIn("resource busy:", failure)
        self.assertIn("of 0.15s", failure)

    def test_build_lock_timeout_uses_max_of_30s_and_device_window_wait_seconds(self):
        recorded_timeouts: list[float] = []
        orig_acquire = engine_mod.acquire

        def _spy_acquire(resources, *, root=None, timeout=0.0):
            recorded_timeouts.append(float(timeout))
            return orig_acquire(resources, root=root, timeout=timeout)

        with patch.object(engine_mod, "acquire", side_effect=_spy_acquire):
            run_default = self.engine.run("test-pack", self.source, ["test"])
        self.assertEqual(run_default["status"], "PASS")
        self.assertEqual(recorded_timeouts, [30.0])

        recorded_timeouts.clear()
        self.manifest["recipes"][0]["deviceWindow"] = {"waitSeconds": 60}
        self._write_manifest()
        with patch.object(engine_mod, "acquire", side_effect=_spy_acquire):
            self.engine.run("test-pack", self.source, ["test"])
        self.assertEqual(recorded_timeouts, [60.0])

        recorded_timeouts.clear()
        self.manifest["recipes"][0]["deviceWindow"] = {"waitSeconds": 10}
        self._write_manifest()
        with patch.object(engine_mod, "acquire", side_effect=_spy_acquire):
            self.engine.run("test-pack", self.source, ["test"])
        self.assertEqual(recorded_timeouts, [30.0])
