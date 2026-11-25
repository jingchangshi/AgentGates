"""Host tests for the tilefoundry-npu L3 promotion and L4 idle policies."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "agent_gates" / "adapters" / "tilefoundry-npu"
SCRIPT = ADAPTER / "scripts" / "tilefoundry_host_gate.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_script(SCRIPT, "tilefoundry_host_gate_l3_test")

AR_TESTS = (
    "test_attention_residual_cannbench_cases",
    "test_attention_residual_baseline_vs_tuned_parity",
    "test_dynamic_shape_kernel_cache_reuse",
)


class _Seen:
    def __init__(self):
        self.commands: list[tuple[str, dict]] = []
        self.workload_env: dict | None = None
        self.cache_root: Path | None = None


class _Fixture:
    """Synthetic TileFoundry checkout, CANNBench checkout, and machine file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "tf-repo"
        self.repo = self.root / "cann-bench"
        self.work_root = self.root / "tf-work"
        self.lock_root = self.root / "locks"
        self.candidate = (
            self.src / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
        )
        self.candidate.parent.mkdir(parents=True)
        self.candidate.write_text("# candidate\n", encoding="utf-8")
        (self.src / "examples" / "ascend950pr_ops" / "cannbench_targets.json").write_text(
            json.dumps([{
                "task": "level4/attention_residual",
                "candidate": "examples/ascend950pr_ops/attention_residual/kernels.py",
            }]),
            encoding="utf-8",
        )
        for sub in ("src/tilefoundry", "tests/ascend/target", "tests/ascend/operators"):
            (self.src / sub).mkdir(parents=True)
        for sub in (
            "scripts",
            "src/kernel_eval",
            "src/cann_bench_utils",
            "tasks/metadata",
            "tasks/level4/attention_residual",
        ):
            (self.repo / sub).mkdir(parents=True)
        (self.repo / "scripts" / "run_evaluation.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.repo / "src" / "cann_bench_utils" / "CMakeLists.txt").write_text(
            "# cmake\n", encoding="utf-8"
        )
        task_dir = self.repo / "tasks" / "level4" / "attention_residual"
        for name in ("proto.yaml", "golden.py"):
            (task_dir / name).write_text(f"# {name}\n", encoding="utf-8")
        (task_dir / "cases.yaml").write_text(
            "cases:\n  - id: '1'\n  - id: '16'\n", encoding="utf-8"
        )
        self.cann = self.root / "set_env.sh"
        self.cann.write_text("#!/bin/sh\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.python = self.root / "python3"
        self.python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.python.chmod(0o755)
        self.env_file = self.root / "machine.env"
        self.env_file.write_text(
            "".join(
                f"{key}={value}\n"
                for key, value in {
                    "TILEFOUNDRY_NPU_CANN_SETUP": str(self.cann),
                    "TILEFOUNDRY_NPU_CONDA_SH": str(self.conda_sh),
                    "TILEFOUNDRY_NPU_CONDA_ENV": "tilefoundry-npu",
                    "TILEFOUNDRY_NPU_PYTHON": str(self.python),
                    "TILEFOUNDRY_NPU_BUILD_SUBDIR": "build",
                    "TILEFOUNDRY_NPU_CANNBENCH_REPO": str(self.repo),
                    "TILEFOUNDRY_NPU_WORK_ROOT": str(self.work_root),
                }.items()
            ),
            encoding="utf-8",
        )
        self.lease = self.lock_root / "npu.lock"

    def environment(self, **overrides):
        env = {
            "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
            "TILEFOUNDRY_NPU_DEVICE_ID": "1",
            "AGENT_GATES_DEVICE_LEASE_PATH": str(self.lease),
            "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
            "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
            "AGENT_GATES_TARGET": "level4/attention_residual",
            "AGENT_GATES_CANDIDATE": "examples/ascend950pr_ops/attention_residual/kernels.py",
        }
        env.update(overrides)
        return env

    def values(self):
        return GATE.machine_values()

    @contextmanager
    def in_source(self):
        previous = os.getcwd()
        os.chdir(self.src)
        try:
            yield
        finally:
            os.chdir(previous)

    def preflight_facts(self, env, seen):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        seen.cache_root = cache_root
        return {
            "ok": True,
            "python": str(self.python.resolve()),
            "cache": {
                name: {"writable": True, "path": str(cache_root / name)}
                for name in ("tilelang", "reference")
            },
            "tilelangCacheDir": str(cache_root / "tilelang"),
            "caseIds": ["1", "16"],
        }

    def workload_facts(self, env, seen):
        seen.workload_env = dict(env)
        return {
            "interpreter": str(self.python.resolve()),
            "sysPathHead": [],
            "tilelangSourceMatches": None,
            "cacheDirs": {},
        }

    def evidence(self, env):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        digest = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        return {
            "schema": "tilefoundry-l3-evidence/v1",
            "profile": "l3-promotion",
            "target": {
                "operator": "attention_residual",
                "task": "level4/attention_residual",
                "taskDir": str(self.repo / "tasks" / "level4" / "attention_residual"),
                "modulePath": str(self.candidate.resolve()),
                "cases": ["1", "16"],
            },
            "candidate": {
                "sha256": digest,
                "modulePath": str(self.candidate.resolve()),
            },
            "interpreter": {
                "match": True,
                "sysExecutable": str(self.python.resolve()),
            },
            "referenceCache": {
                "mode": "required",
                "required": True,
                "usable": True,
                "expectation": env.get("TF_L3_CACHE_EXPECTATION"),
                "hits": 3,
                "misses": 2,
                "writes": 2,
                "errors": [],
                "byKind": {},
                "location": str(cache_root / "reference"),
            },
            "coverage": {
                "qualified": True,
                "referenceCacheProblem": None,
                "nodes": {
                    "expected": 3,
                    "completed": 3,
                    "missing": [],
                    "failed": [],
                    "skipped": [],
                },
                "cases": {
                    "expected": 20,
                    "completed": 20,
                    "missing": [],
                    "unexpected": [],
                },
            },
            "timing": {
                "startedAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-01T00:01:00Z",
                "finishedAt": "2026-01-01T00:01:00Z",
            },
            "phase": "finished",
            "writeErrors": [],
            "interruption": {
                "interrupted": False,
                "errorCategory": None,
                "signal": None,
            },
        }

    def junit_xml(self, *, statuses=None):
        statuses = statuses or {name: "passed" for name in AR_TESTS}
        rows = []
        for name, status in statuses.items():
            inner = ""
            if status == "failed":
                inner = "<failure message='wrong'>mismatch</failure>"
            elif status == "skipped":
                inner = "<skipped message='skip'/>"
            rows.append(
                f"<testcase name='{name}' classname='test_npu_acceptance' "
                f"time='1.5'>{inner}</testcase>"
            )
        failed = sum(1 for status in statuses.values() if status == "failed")
        skipped = sum(1 for status in statuses.values() if status == "skipped")
        return (
            f"<testsuite tests='{len(statuses)}' errors='0' failures='{failed}' "
            f"skipped='{skipped}'>" + "".join(rows) + "</testsuite>"
        )

    @staticmethod
    def _cache_problem(cache):
        """Mirror the producer's reference-cache policy at the evidence boundary."""
        if bool(cache.get("required")) and not bool(cache.get("usable")):
            return "reference_cache_unavailable"
        if bool(cache.get("required")) and cache.get("errors"):
            return "reference_cache_error"
        if cache.get("policyChanged"):
            return "reference_cache_policy_changed"
        expectation = str(cache.get("expectation") or "any")
        by_kind = cache.get("byKind") if isinstance(cache.get("byKind"), dict) else {}
        oracle = by_kind.get("oracle") if isinstance(by_kind.get("oracle"), dict) else {}
        hits = int(oracle.get("hits", 0))
        misses = int(oracle.get("misses", 0))
        writes = int(oracle.get("writes", 0))
        if expectation == "cold" and (misses <= 0 or writes != misses):
            return "reference_cache_expectation_cold"
        if expectation == "warm" and (misses != 0 or writes != 0 or hits <= 0):
            return "reference_cache_expectation_warm"
        return ""

    def cache_evidence(self, *, hits, misses, writes, by_kind, errors=None, usable=True):
        """Build evidence whose cache policy matches the producer's observation."""
        base = self.evidence

        def build(env):
            packet = json.loads(json.dumps(base(env)))
            cache = packet["referenceCache"]
            cache.update({
                "mode": "required",
                "required": True,
                "usable": usable,
                "expectation": env.get("TF_L3_CACHE_EXPECTATION"),
                "hits": hits,
                "misses": misses,
                "writes": writes,
                "errors": list(errors or ()),
                "byKind": json.loads(json.dumps(by_kind)),
            })
            problem = self._cache_problem(cache)
            packet["coverage"]["referenceCacheProblem"] = problem or None
            packet["coverage"]["qualified"] = not problem
            packet["interruption"]["errorCategory"] = problem or None
            return packet

        return build

    @staticmethod
    def producer_failure(
        *,
        kind,
        category,
        stage,
        node=None,
        case=None,
        role=None,
        comparator=None,
        references=(),
        comparisons=(),
        exception_type=None,
        detail="",
    ):
        """Build the exact structured failure shared by the producer and consumers."""
        return {
            "kind": kind,
            "category": category,
            "stage": stage,
            "node": node,
            "case": None if case is None else str(case),
            "role": role,
            "comparator": comparator,
            "references": [str(item) for item in references],
            "comparisons": [
                {
                    "output": row.get("output"),
                    "passed": bool(row.get("passed")),
                    "message": str(row.get("message") or ""),
                }
                for row in comparisons
            ],
            "exceptionType": exception_type,
            "detail": detail,
        }

    @staticmethod
    def _top_failure(failures):
        """Mirror the producer's infrastructure > coverage > candidate precedence."""
        priority = {"infrastructure": 0, "coverage": 1, "candidate": 2}
        selected = None
        selected_priority = 0
        for row in failures:
            rank = priority.get(str(row.get("kind") or ""), 99)
            if selected is None or rank < selected_priority:
                selected = row
                selected_priority = rank
        return selected

    def failure_evidence(self, failures, *, node_states=None):
        """Publish failures exactly as the producer does: top, list, node, case."""
        base = self.evidence
        failures = [json.loads(json.dumps(row)) for row in failures]

        def build(env):
            packet = json.loads(json.dumps(base(env)))
            by_node = {}
            for row in failures:
                node_name = row.get("node")
                if isinstance(node_name, str) and node_name:
                    by_node.setdefault(node_name, []).append(row)
            nodes = []
            cases = {}
            for node_name, rows in by_node.items():
                states = dict((node_states or {}).get(node_name) or {})
                if not states:
                    case_id = next(
                        (str(row["case"]) for row in rows if row.get("case") is not None),
                        "1",
                    )
                    states = {case_id: "failed"}
                node_failure = self._top_failure(rows)
                nodes.append({
                    "name": node_name,
                    "cases": states,
                    "failure": json.loads(json.dumps(node_failure)) if node_failure else None,
                })
                for case_id, status in states.items():
                    case_rows = [
                        row for row in rows if str(row.get("case")) == str(case_id)
                    ]
                    case_failure = self._top_failure(case_rows) if case_rows else None
                    cases[f"{node_name}:{case_id}"] = {
                        "status": status,
                        "failure": (
                            json.loads(json.dumps(case_failure)) if case_failure else None
                        ),
                    }
            top = self._top_failure(failures) if failures else None
            packet["failure"] = json.loads(json.dumps(top)) if top else None
            packet["failures"] = failures
            packet["nodes"] = nodes
            packet["cases"] = cases
            return packet

        return build

    def candidate_failure(self, *, node=AR_TESTS[0], case="1"):
        return self.producer_failure(
            kind="candidate",
            category="numeric",
            stage="comparison",
            node=node,
            case=case,
            role="candidate",
            comparator="cannbench",
            references=("oracle", "native"),
            comparisons=(
                {"output": 0, "passed": False, "message": "mismatch"},
            ),
            detail="numeric mismatch",
        )

    def numeric_evidence(self, *, node=AR_TESTS[0], case="1"):
        return self.failure_evidence([self.candidate_failure(node=node, case=case)])

    def fake_runner(self, facts_writer=None, workload_writer=None, preflight_rc=0, workload_rc=0):
        """Return a run_host_bash side effect that writes plausible facts."""
        seen = _Seen()

        def run(cmd, *, cwd, env, **kwargs):
            seen.commands.append((cmd, dict(env)))
            args = cmd.split()
            script = Path(args[2]).name
            if script == "preflight.py":
                facts_path = Path(args[3])
                facts = (facts_writer or self.preflight_facts)(env, seen)
                facts_path.write_text(json.dumps(facts), encoding="utf-8")
                return types.SimpleNamespace(
                    returncode=preflight_rc, stdout="", stderr=""
                )
            facts_path = Path(args[3])
            report = Path(args[4])
            facts = (workload_writer or self.workload_facts)(env, seen)
            facts_path.write_text(json.dumps(facts), encoding="utf-8")
            report.write_text(self.junit_xml(), encoding="utf-8")
            evidence_path = Path(env["TF_L3_EVIDENCE_PATH"])
            evidence_path.write_text(json.dumps(self.evidence(env)), encoding="utf-8")
            return types.SimpleNamespace(returncode=workload_rc, stdout="", stderr="")

        run.seen = seen  # type: ignore[attr-defined]
        return run


class L3PromotionCase(_Fixture, unittest.TestCase):
    def run_l3(self, runner=None, *, env=None):
        runner = runner or self.fake_runner()
        with patch.dict(os.environ, env or self.environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=runner), \
                self.in_source():
            return GATE.run_l3_promotion(self.values(), 60.0)

    def test_request_pair_is_required_and_rejects_partial_input(self):
        with patch.dict(
            os.environ,
            {"AGENT_GATES_TARGET": "level4/attention_residual"},
            clear=True,
        ):
            with self.assertRaises(GATE.BridgeError):
                GATE._read_l3_request()
        with patch.dict(
            os.environ,
            {
                "AGENT_GATES_TARGET": "level4/attention_residual",
                "AGENT_GATES_CANDIDATE": (
                    "examples/ascend950pr_ops/attention_residual/kernels.py"
                ),
            },
            clear=True,
        ):
            self.assertEqual(
                GATE._read_l3_request(),
                {
                    "task": "level4/attention_residual",
                    "candidate": (
                        "examples/ascend950pr_ops/attention_residual/kernels.py"
                    ),
                },
            )

    def test_cold_run_promotes_with_dedicated_caches(self):
        shared = self.root / "shared-cache"
        shared.mkdir()
        sentinel = shared / "keep.bin"
        sentinel.write_bytes(b"shared")
        runner = self.fake_runner()
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 0, obs)
        self.assertTrue(obs["promotionEligible"])
        self.assertTrue(obs["coverageComplete"])
        self.assertTrue(obs["cachePreflightOk"])
        self.assertTrue(obs["candidateDigestRecorded"])
        self.assertEqual(obs["testsCollected"], 3)
        self.assertEqual(obs["testsFailed"], 0)
        self.assertEqual(obs["caseCount"], 20)
        self.assertEqual(obs["passedCases"], 20)
        self.assertEqual(obs["errorCategory"], "none")
        self.assertIsNone(obs["failureCategory"])
        self.assertEqual(
            obs["evidence"]["target"]["modulePath"], str(self.candidate.resolve())
        )
        self.assertEqual(
            obs["evidence"]["candidate"]["modulePath"], str(self.candidate.resolve())
        )
        self.assertEqual(
            obs["evidence"]["candidate"]["relativePath"],
            "examples/ascend950pr_ops/attention_residual/kernels.py",
        )
        self.assertEqual(
            obs["evidence"]["candidate"]["sha256"],
            hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
        )
        expected = obs["evidence"]["expectedIdentity"]
        self.assertEqual(expected["target"]["modulePath"], str(self.candidate.resolve()))
        self.assertEqual(expected["candidate"]["modulePath"], str(self.candidate.resolve()))
        self.assertEqual(
            expected["candidate"]["relativePath"],
            "examples/ascend950pr_ops/attention_residual/kernels.py",
        )
        self.assertEqual(
            expected["candidate"]["sha256"],
            hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
        )
        # An empty dedicated cache is still an ordinary promotion run. The
        # strict cold/warm judgement belongs to the Lead's explicit acceptance
        # run; ordinary promotion accepts the legal cold rebuild and records
        # the per-kind counters for that independent check.
        self.assertEqual(obs["cachePolicy"]["expectation"], "any")
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        self.assertFalse((cache_root / "reference-warm.json").exists())
        workload_env = runner.seen.workload_env
        assert workload_env is not None
        self.assertEqual(workload_env["TF_L3_EVIDENCE_PATH"], str(obs["paths"]["evidence"]))
        self.assertEqual(workload_env["TF_L3_TARGET_OPERATOR"], "level4/attention_residual")
        self.assertEqual(
            workload_env["TF_L3_TARGET_MODULE_PATH"], str(self.candidate.resolve())
        )
        self.assertEqual(
            workload_env["TF_L3_EXPECTED_PYTHON"], str(self.python.resolve())
        )
        self.assertEqual(workload_env["CANNBENCH_CACHE_MODE"], "required")
        self.assertEqual(workload_env["TILELANG_CACHE_DIR"], str(cache_root / "tilelang"))
        self.assertNotIn("AGENT_GATES_TARGET", workload_env)
        self.assertNotIn("AGENT_GATES_CANDIDATE", workload_env)
        self.assertEqual(sentinel.read_bytes(), b"shared")
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        self.assertEqual(obs["deviceWindow"]["occupancy"], "free")

    def test_missing_producer_identity_blocks_promotion_without_backfill(self):
        original = self.evidence
        mutations = {
            "target.modulePath": lambda packet: packet["target"].pop("modulePath"),
            "candidate.modulePath": lambda packet: packet["candidate"].pop("modulePath"),
            "candidate.sha256": lambda packet: packet["candidate"].pop("sha256"),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                def without_field(env, mutate=mutate):
                    packet = json.loads(json.dumps(original(env)))
                    mutate(packet)
                    return packet

                self.evidence = without_field  # type: ignore[method-assign]
                rc, obs = self.run_l3()
                self.assertEqual(rc, 1, obs)
                self.assertFalse(obs["promotionEligible"])
                self.assertFalse(obs["coverageComplete"])
                self.assertEqual(obs["failureCategory"], "environment")
                expected = obs["evidence"]["expectedIdentity"]
                self.assertEqual(
                    expected["target"]["modulePath"], str(self.candidate.resolve())
                )
                self.assertEqual(
                    expected["candidate"]["modulePath"], str(self.candidate.resolve())
                )
                self.assertEqual(
                    expected["candidate"]["sha256"],
                    hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
                )
                if field == "target.modulePath":
                    self.assertIsNone(obs["evidence"]["target"]["modulePath"])
                elif field == "candidate.modulePath":
                    self.assertNotIn("modulePath", obs["evidence"]["candidate"])
                else:
                    self.assertNotIn("sha256", obs["evidence"]["candidate"])

    def test_a_foreign_producer_candidate_path_blocks_promotion(self):
        original = self.evidence
        foreign = str(self.root / "other-candidate.py")

        def with_foreign_candidate(env):
            packet = json.loads(json.dumps(original(env)))
            packet["candidate"]["modulePath"] = foreign
            return packet

        self.evidence = with_foreign_candidate  # type: ignore[method-assign]
        rc, obs = self.run_l3()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["failureCategory"], "environment")
        self.assertEqual(obs["evidence"]["candidate"]["modulePath"], foreign)

    def test_a_foreign_producer_target_path_blocks_promotion(self):
        original = self.evidence
        foreign = str(self.root / "other-target.py")

        def with_foreign_target(env):
            packet = json.loads(json.dumps(original(env)))
            packet["target"]["modulePath"] = foreign
            return packet

        self.evidence = with_foreign_target  # type: ignore[method-assign]
        rc, obs = self.run_l3()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["failureCategory"], "environment")
        self.assertEqual(obs["evidence"]["target"]["modulePath"], foreign)

    def test_ordinary_promotion_does_not_force_warm_from_a_marker(self):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        reference = cache_root / "reference" / "operators"
        reference.mkdir(parents=True)
        (reference / "entry.json").write_text("{}", encoding="utf-8")
        runner = self.fake_runner()
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 0, obs)
        self.assertEqual(obs["cachePolicy"]["expectation"], "any")
        self.assertEqual(runner.seen.workload_env["TF_L3_CACHE_EXPECTATION"], "any")

        (cache_root / "reference-warm.json").write_text(
            json.dumps({
                "task": "level4/attention_residual",
                "operator": "attention_residual",
            }),
            encoding="utf-8",
        )
        runner2 = self.fake_runner()
        rc2, obs2 = self.run_l3(runner2)
        self.assertEqual(rc2, 0, obs2)
        # A marker is not a strict warm-acceptance request. Ordinary promotion
        # must still tolerate a legitimate rebuild; the acceptance run opts in
        # to cold/warm policy separately.
        self.assertEqual(obs2["cachePolicy"]["expectation"], "any")
        self.assertEqual(runner2.seen.workload_env["TF_L3_CACHE_EXPECTATION"], "any")

    def test_ordinary_promotion_accepts_mixed_cache_with_a_marker(self):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        reference = cache_root / "reference" / "operators"
        reference.mkdir(parents=True)
        (reference / "entry.json").write_text("{}", encoding="utf-8")
        (cache_root / "reference-warm.json").write_text(
            json.dumps({
                "task": "level4/attention_residual",
                "operator": "attention_residual",
                # A real marker carries the old run's candidate and counters.
                # Neither may turn a legitimate identity rebuild into a strict
                # warm acceptance run.
                "candidateSha256": "0" * 64,
                "referenceCache": {
                    "hits": 9,
                    "misses": 0,
                    "writes": 0,
                    "byKind": {"oracle": {"hits": 9, "misses": 0, "writes": 0}},
                },
                "at": "2026-01-01T00:00:00+00:00",
            }),
            encoding="utf-8",
        )
        by_kind = {
            "oracle": {"hits": 2, "misses": 2, "writes": 2},
            "native": {"hits": 0, "misses": 0, "writes": 0},
        }
        # A changed ReferenceCache.identity rebuilds the operator folder. Those
        # misses/writes are a legitimate ordinary rebuild, not a strict warm
        # acceptance run, so the marker must not turn them into a cache block.
        self.evidence = self.cache_evidence(
            hits=2, misses=2, writes=2, by_kind=by_kind,
        )
        runner = self.fake_runner()
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 0, obs)
        self.assertTrue(obs["promotionEligible"])
        self.assertTrue(obs["coverageComplete"])
        self.assertEqual(obs["cachePolicy"]["expectation"], "any")
        self.assertEqual(runner.seen.workload_env["TF_L3_CACHE_EXPECTATION"], "any")
        self.assertIsNone(obs["coverage"]["referenceCacheProblem"])
        self.assertTrue(obs["coverage"]["qualified"])
        self.assertIsNone(obs["failureCategory"])
        self.assertEqual(obs["referenceCache"]["byKind"], by_kind)
        self.assertEqual(
            (obs["referenceCache"]["hits"], obs["referenceCache"]["misses"],
             obs["referenceCache"]["writes"]),
            (2, 2, 2),
        )

    def test_ordinary_promotion_blocks_a_reference_cache_write_failure_with_a_marker(self):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        reference = cache_root / "reference" / "operators"
        reference.mkdir(parents=True)
        (reference / "entry.json").write_text("{}", encoding="utf-8")
        (cache_root / "reference-warm.json").write_text(
            json.dumps({
                "task": "level4/attention_residual",
                "operator": "attention_residual",
            }),
            encoding="utf-8",
        )
        by_kind = {
            "oracle": {"hits": 1, "misses": 1, "writes": 0},
            "native": {"hits": 0, "misses": 0, "writes": 0},
        }
        error = "reference cache write failed for 1 oracle: OSError: disk full"
        self.evidence = self.cache_evidence(
            hits=1, misses=1, writes=0, by_kind=by_kind, errors=[error],
        )
        runner = self.fake_runner()
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 1, obs)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["failureCategory"], "environment")
        self.assertEqual(obs["coverage"]["referenceCacheProblem"], "reference_cache_error")
        self.assertEqual(obs["referenceCache"]["writes"], 0)
        self.assertEqual(
            obs["referenceCache"]["byKind"]["oracle"],
            {"hits": 1, "misses": 1, "writes": 0},
        )
        self.assertIn(error, obs["referenceCache"]["errors"])

    def test_signed_cache_summary_preserves_per_kind_counts(self):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        reference = cache_root / "reference" / "operators"
        reference.mkdir(parents=True)
        (reference / "entry.json").write_text("{}", encoding="utf-8")
        by_kind = {
            "oracle": {"hits": 4, "misses": 1, "writes": 1},
            "native": {"hits": 3, "misses": 2, "writes": 2},
        }
        # This observation is what the engine signs into the receipt. Keep the
        # per-kind counters intact so the Lead can verify cold/warm independently.
        self.evidence = self.cache_evidence(
            hits=7, misses=3, writes=3, by_kind=by_kind,
        )
        rc, obs = self.run_l3(self.fake_runner())
        self.assertEqual(rc, 0, obs)
        self.assertEqual(obs["cachePolicy"]["expectation"], "any")
        self.assertEqual(obs["referenceCache"]["byKind"], by_kind)
        self.assertEqual(obs["evidence"]["referenceCache"]["byKind"], by_kind)

    def test_stale_or_truncated_marker_is_not_treated_as_warm(self):
        cache_root = self.work_root / "l3" / "cache" / "attention_residual"
        reference = cache_root / "reference" / "operators"
        reference.mkdir(parents=True)
        (reference / "entry.json").write_text("{}", encoding="utf-8")
        marker = cache_root / "reference-warm.json"
        marker.write_text('{"task": "level4/attention_residual"', encoding="utf-8")
        runner = self.fake_runner()
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 0, obs)
        self.assertEqual(obs["cachePolicy"]["expectation"], "any")
        self.assertEqual(runner.seen.workload_env["TF_L3_CACHE_EXPECTATION"], "any")

        marker.write_text(
            json.dumps({"task": "level4/other", "operator": "attention_residual"}),
            encoding="utf-8",
        )
        runner2 = self.fake_runner()
        rc2, obs2 = self.run_l3(runner2)
        self.assertEqual(rc2, 0, obs2)
        self.assertEqual(obs2["cachePolicy"]["expectation"], "any")

    def test_missing_or_all_skipped_tests_never_promote(self):
        original = self.junit_xml

        def skip_two(**kwargs):
            return original(statuses={
                AR_TESTS[0]: "skipped",
                AR_TESTS[1]: "skipped",
            })

        self.junit_xml = skip_two  # type: ignore[method-assign]
        rc, obs = self.run_l3()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertFalse(obs["notAllSkipped"])
        self.assertEqual(obs["errorCategory"], "coverage")
        self.assertEqual(obs["failureCategory"], "coverage")
        self.assertFalse(
            (self.work_root / "l3/cache/attention_residual/reference-warm.json").exists()
        )

    def test_numeric_failure_is_returned_as_candidate_evidence(self):
        original = self.junit_xml

        def first_failed(**kwargs):
            return original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = self.numeric_evidence()
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "numeric")
        self.assertEqual(obs["failureCategory"], "numeric")
        self.assertEqual(obs["testsFailed"], 1)
        self.assertEqual(obs["testsCollected"], 3)
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])

    def test_write_errors_on_a_green_run_are_insufficient_evidence(self):
        original = self.evidence
        errors = ["evidence.json: [Errno 28] No space left on device"]

        def with_write_errors(env):
            packet = json.loads(json.dumps(original(env)))
            packet["writeErrors"] = list(errors)
            return packet

        self.evidence = with_write_errors  # type: ignore[method-assign]
        rc, obs = self.run_l3()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertEqual(obs["failureCategory"], "environment")
        self.assertEqual(obs["evidence"]["writeErrors"], errors)
        self.assertEqual(obs["testsFailed"], 0)

    def test_write_errors_do_not_mask_a_real_numeric_failure(self):
        """Producer write errors decide the verdict; numeric stays diagnostic.

        The producer records the observed candidate mismatch as one failure row
        and appends an ``infrastructure`` gap row for ``writeErrors``. The top
        failure is therefore infrastructure: the run must pause, remain pending
        and never promote, while the numeric row is kept in ``failures`` as
        diagnostic evidence.
        """
        candidate = self.candidate_failure()
        errors = ["final evidence write failed"]
        write_error_failure = self.producer_failure(
            kind="infrastructure",
            category="unclassified",
            stage="report",
            detail=errors[0],
        )
        producer_evidence = self.failure_evidence([candidate, write_error_failure])

        def with_write_errors(env):
            packet = producer_evidence(env)
            packet["writeErrors"] = list(errors)
            return packet

        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = with_write_errors  # type: ignore[method-assign]
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertEqual(obs["failureCategory"], "environment")
        payload = obs["evidence"]
        self.assertEqual(payload["writeErrors"], errors)
        self.assertEqual(payload["failure"]["kind"], "infrastructure")
        self.assertEqual(payload["failure"]["category"], "unclassified")
        self.assertEqual(payload["failure"]["stage"], "report")
        self.assertIsNone(payload["failure"]["case"])
        self.assertEqual(
            [row["kind"] for row in payload["failures"]],
            ["candidate", "infrastructure"],
        )
        diagnostic = payload["failures"][0]
        self.assertEqual(diagnostic["category"], "numeric")
        self.assertEqual(diagnostic["role"], "candidate")
        self.assertEqual(diagnostic["stage"], "comparison")
        self.assertEqual(diagnostic["case"], "1")
        self.assertTrue(
            any(row["passed"] is False for row in diagnostic["comparisons"])
        )

    def test_write_errors_without_a_final_snapshot_are_environment(self):
        """A missing final snapshot is insufficient evidence, even with a marker.

        The producer may have recorded the candidate comparison before its last
        write failed. The on-disk payload is then non-terminal; it may inform
        diagnostics but cannot become a numeric verdict or advance stagnation.
        """
        producer_evidence = self.numeric_evidence()
        errors = ["final evidence write failed"]

        def incomplete_snapshot(env):
            packet = producer_evidence(env)
            packet["writeErrors"] = list(errors)
            packet["phase"] = "case:attention_residual:1"
            packet["timing"]["finishedAt"] = None
            return packet

        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = incomplete_snapshot  # type: ignore[method-assign]
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertEqual(obs["failureCategory"], "environment")
        self.assertNotEqual(obs["failureCategory"], "numeric")
        self.assertEqual(obs["evidence"]["writeErrors"], errors)
        self.assertEqual(obs["evidence"]["phase"], "case:attention_residual:1")

    def test_plain_junit_failure_without_producer_marker_is_not_numeric(self):
        original = self.junit_xml

        def first_failed(**kwargs):
            return original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "infra")
        self.assertEqual(obs["failureCategory"], "infra")
        self.assertNotEqual(obs["errorCategory"], "numeric")

    def test_numeric_marker_for_a_different_node_is_not_accepted(self):
        junit_original = self.junit_xml
        self.evidence = self.numeric_evidence(node=AR_TESTS[1])

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "infra")
        self.assertNotEqual(obs["errorCategory"], "numeric")

    def test_producer_infrastructure_failure_is_not_numeric(self):
        junit_original = self.junit_xml
        failure = self.producer_failure(
            kind="infrastructure",
            category="exception",
            stage="kernel",
            node=AR_TESTS[0],
            case="1",
            exception_type="RuntimeError",
            detail="kernel RuntimeError: device unavailable",
        )

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = self.failure_evidence([failure])
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertEqual(obs["failureCategory"], "environment")
        payload_failure = obs["evidence"]["failure"]
        self.assertEqual(payload_failure["kind"], "infrastructure")
        self.assertEqual(payload_failure["category"], "exception")
        self.assertEqual(payload_failure["stage"], "kernel")
        self.assertEqual(payload_failure["exceptionType"], "RuntimeError")
        self.assertEqual(payload_failure["case"], "1")

    def test_producer_coverage_failure_is_not_numeric(self):
        junit_original = self.junit_xml
        failure = self.producer_failure(
            kind="coverage",
            category="missing_case",
            stage="coverage",
            node=AR_TESTS[0],
            case="16",
            detail="case 16 was not completed",
        )

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = self.failure_evidence(
            [failure], node_states={AR_TESTS[0]: {"1": "passed"}},
        )
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "coverage")
        self.assertEqual(obs["failureCategory"], "coverage")
        payload_failure = obs["evidence"]["failure"]
        self.assertEqual(payload_failure["kind"], "coverage")
        self.assertEqual(payload_failure["category"], "missing_case")
        self.assertEqual(payload_failure["case"], "16")

    def test_same_call_phase_failure_uses_the_producer_shape(self):
        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        # Both runs see the same pytest call-phase <failure> row. Only the
        # structured producer evidence distinguishes execution from a numeric
        # candidate verdict.
        shared_junit = first_failed()
        self.assertIn("<failure", shared_junit)
        cases = (
            (
                self.producer_failure(
                    kind="infrastructure",
                    category="exception",
                    stage="kernel",
                    node=AR_TESTS[0],
                    case="1",
                    exception_type="RuntimeError",
                    detail="kernel RuntimeError: device unavailable",
                ),
                "environment",
                "infrastructure",
            ),
            (
                self.candidate_failure(),
                "numeric",
                "candidate",
            ),
        )
        for failure, expected_category, expected_kind in cases:
            with self.subTest(kind=expected_kind):
                self.evidence = self.failure_evidence([failure])
                self.junit_xml = lambda **_: shared_junit
                rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
                self.assertEqual(rc, 1)
                self.assertFalse(obs["promotionEligible"])
                self.assertEqual(obs["failureCategory"], expected_category)
                self.assertEqual(obs["evidence"]["failure"]["kind"], expected_kind)
                if expected_kind == "candidate":
                    self.assertEqual(obs["evidence"]["failure"]["category"], "numeric")
                else:
                    self.assertNotEqual(obs["failureCategory"], "numeric")

    def test_infrastructure_top_failure_precedes_a_candidate_numeric_row(self):
        junit_original = self.junit_xml
        candidate = self.candidate_failure()
        infrastructure = self.producer_failure(
            kind="infrastructure",
            category="exception",
            stage="kernel",
            node=AR_TESTS[0],
            case="1",
            exception_type="RuntimeError",
            detail="kernel RuntimeError: device unavailable",
        )

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = self.failure_evidence([candidate, infrastructure])
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["failureCategory"], "environment")
        top = obs["evidence"]["failure"]
        self.assertEqual(top["kind"], "infrastructure")
        self.assertEqual(top["category"], "exception")
        self.assertEqual(top["stage"], "kernel")
        self.assertEqual(
            [row["kind"] for row in obs["evidence"]["failures"]],
            ["candidate", "infrastructure"],
        )
        diagnostic = obs["evidence"]["failures"][0]
        self.assertEqual(diagnostic["category"], "numeric")
        self.assertEqual(diagnostic["role"], "candidate")
        self.assertNotEqual(obs["failureCategory"], "numeric")

    def test_a_stale_unfinalized_snapshot_is_not_a_numeric_failure(self):
        original = self.evidence

        def stale_snapshot(env):
            packet = json.loads(json.dumps(original(env)))
            packet["phase"] = "case:attention_residual:1"
            packet["timing"]["finishedAt"] = None
            packet.pop("writeErrors")
            return packet

        self.evidence = stale_snapshot  # type: ignore[method-assign]
        rc, obs = self.run_l3()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertEqual(obs["failureCategory"], "environment")
        self.assertEqual(obs["evidence"]["writeErrors"], [])
        self.assertEqual(obs["testsFailed"], 0)

    def test_l3_uses_importable_comparator_paths_and_creates_no_shim(self):
        runner = self.fake_runner()
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 0, obs)
        run_dir = Path(obs["paths"]["runDir"])
        self.assertFalse((run_dir / "shim").exists())
        self.assertFalse(any(run_dir.rglob("npu_compare.py")))
        self.assertFalse(hasattr(GATE, "NPU_COMPARE_SHIM"))
        self.assertIn("from kernel_eval.utils.compare import compare_tensors", GATE.L3_PREFLIGHT)
        for _, env in runner.seen.commands:
            parts = env["PYTHONPATH"].split(":")
            self.assertEqual(parts[0], str(self.src.resolve()))
            self.assertIn(str(self.repo.resolve() / "src"), parts)
            self.assertNotIn(str(run_dir / "shim"), parts)

    def test_cache_preflight_failure_blocks_before_the_device_workload(self):
        def bad_facts(env, seen):
            facts = self.preflight_facts(env, seen)
            facts["cache"]["reference"] = {
                "writable": False,
                "path": str(self.work_root / "l3/cache/attention_residual/reference"),
            }
            return facts

        runner = self.fake_runner(facts_writer=bad_facts)
        rc, obs = self.run_l3(runner)
        self.assertEqual(rc, 75)
        self.assertFalse(obs["cachePreflightOk"])
        self.assertFalse(obs["windowOpened"])
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertIsNone(runner.seen.workload_env)

    def test_workload_timeout_preserves_partial_evidence(self):
        def timeout_on_workload(cmd, *, cwd, env, **kwargs):
            if Path(cmd.split()[2]).name == "preflight.py":
                facts_path = Path(cmd.split()[3])
                facts_path.write_text(
                    json.dumps(self.preflight_facts(env, _Seen())), encoding="utf-8"
                )
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            raise subprocess.TimeoutExpired(cmd="pytest", timeout=1.0)

        with patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(
                    GATE.adapter_runtime, "run_host_bash", side_effect=timeout_on_workload
                ), \
                self.in_source():
            rc, obs = GATE.run_l3_promotion(self.values(), 60.0)
        self.assertEqual(rc, 1)
        self.assertEqual(obs["errorCategory"], "deadline")
        self.assertIsNone(obs["testsCollected"])
        self.assertIsNone(obs["testsFailed"])
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        phases = {phase["name"]: phase for phase in obs["phases"]}
        self.assertEqual(phases["device-workload"]["status"], "failed")
        self.assertEqual(phases["preflight"]["status"], "ok")

    def test_runtime_health_abort_preserves_partial_evidence_and_health_snapshot(self):
        def health_abort(cmd, **kwargs):
            env = kwargs["env"]
            facts_path = Path(cmd.split()[3])
            report = Path(cmd.split()[4])
            facts_path.write_text(
                json.dumps(self.workload_facts(env, _Seen())), encoding="utf-8"
            )
            report.write_text(self.junit_xml(), encoding="utf-8")
            Path(env["TF_L3_EVIDENCE_PATH"]).write_text(
                json.dumps(self.evidence(env)), encoding="utf-8"
            )
            kwargs["monitor_sink"].update({
                "abort": {
                    "kind": "device_error",
                    "reason": "device health reports 'ERROR'",
                    "phase": "device-workload",
                    "device": "1",
                    "npuId": 7,
                    "healthUnknown": False,
                },
                "cleanup": {
                    "attempted": True,
                    "incomplete": False,
                    "workerJoined": True,
                },
                "outcome": "aborted",
            })
            raise GATE.BridgeError(
                "runtime health abort (device_error): NPU 7 reports ERROR",
                category="environment",
            )

        runner = self.fake_runner()
        with patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=runner), \
                patch.object(GATE, "_run_host_bash_monitored", side_effect=health_abort), \
                self.in_source():
            rc, obs = GATE.run_l3_promotion(self.values(), 60.0)
        self.assertEqual(rc, 1)
        self.assertEqual(obs["errorCategory"], "environment")
        self.assertFalse(obs["promotionEligible"])
        self.assertFalse(obs["coverageComplete"])
        self.assertEqual(obs["healthMonitoring"]["abort"]["kind"], "device_error")
        self.assertEqual(obs["healthMonitoring"]["abort"]["npuId"], 7)
        self.assertEqual(obs["healthMonitoring"]["outcome"], "aborted")
        self.assertTrue(obs["healthMonitoring"]["cleanup"]["workerJoined"])
        self.assertEqual(
            obs["evidence"]["candidate"]["modulePath"], str(self.candidate.resolve())
        )
        self.assertEqual(
            obs["evidence"]["candidate"]["sha256"],
            hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
        )
        self.assertEqual(obs["testsCollected"], 3)
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        phases = {phase["name"]: phase for phase in obs["phases"]}
        self.assertEqual(phases["device-workload"]["status"], "failed")

    def test_exclusive_shared_device_blocks_with_occupancy_evidence(self):
        env = self.environment(TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE="1")
        runner = self.fake_runner()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(
                    GATE.adapter_runtime, "processes_on", return_value=([123], 1)
                ), \
                patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=runner), \
                self.in_source():
            rc, obs = GATE.run_l3_promotion(self.values(), 60.0)
        self.assertEqual(rc, 75)
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertTrue(obs["occupancyRecorded"])
        self.assertIsNone(runner.seen.workload_env)

    def test_l3_blocks_on_shared_device_without_exclusive_flag(self):
        env = self.environment(TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE="0")
        runner = self.fake_runner()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(
                    GATE.adapter_runtime, "processes_on", return_value=([123], 1)
                ), \
                patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=runner), \
                self.in_source():
            rc, obs = GATE.run_l3_promotion(self.values(), 60.0)
        self.assertEqual(rc, 75)
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertFalse(obs["windowOpened"])
        self.assertIsNone(runner.seen.workload_env)

    def test_l3_mid_run_external_occupancy_returns_75_not_1(self):
        def occupancy_abort(cmd, **kwargs):
            kwargs["monitor_sink"].update({
                "abort": {
                    "kind": "external_occupancy",
                    "reason": "L3 lost exclusivity after the lease opened",
                    "phase": "device-workload",
                    "device": "1",
                    "npuId": 7,
                    "occupancy": "shared(1)",
                },
                "outcome": "aborted",
            })
            raise GATE.BridgeError(
                "runtime health abort (external_occupancy)",
                category="occupancy",
                occupancy="shared(1)",
            )

        runner = self.fake_runner()
        with patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=runner), \
                patch.object(GATE, "_run_host_bash_monitored", side_effect=occupancy_abort), \
                self.in_source():
            rc, obs = GATE.run_l3_promotion(self.values(), 60.0)
        self.assertEqual(rc, 75)
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["failureCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])

    def _cleanup_proc(self, proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def _run_l3_shared_quiet_with_monitor(self, *, probe_mem_sequence, workload_sleep):
        ext_proc = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.addCleanup(self._cleanup_proc, ext_proc)
        seen = _Seen()
        probe_mems_seen = []

        def probe_processes(_npu_id, _timeout):
            idx = min(len(probe_mems_seen), len(probe_mem_sequence) - 1)
            mem_mb = probe_mem_sequence[idx]
            probe_mems_seen.append(mem_mb)
            return {
                "ok": True,
                "processTableSeen": True,
                "processes": [{
                    "pid": str(ext_proc.pid),
                    "name": "sleep",
                    "memoryMb": str(mem_mb),
                }],
                "error": None,
            }

        def run_host_bash(cmd, *, cwd, env, **kwargs):
            seen.commands.append((cmd, dict(env)))
            args = cmd.split()
            script = Path(args[2]).name
            if script == "preflight.py":
                Path(args[3]).write_text(
                    json.dumps(self.preflight_facts(env, seen)), encoding="utf-8"
                )
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            completed = subprocess.run(
                ["/bin/sleep", workload_sleep], env=env, check=False
            )
            if completed.returncode == 0:
                Path(args[3]).write_text(
                    json.dumps(self.workload_facts(env, seen)), encoding="utf-8"
                )
                Path(args[4]).write_text(self.junit_xml(), encoding="utf-8")
                Path(env["TF_L3_EVIDENCE_PATH"]).write_text(
                    json.dumps(self.evidence(env)), encoding="utf-8"
                )
            return types.SimpleNamespace(
                returncode=completed.returncode, stdout="", stderr=""
            )

        env = self.environment()
        admission_procs = [
            {"pid": str(ext_proc.pid), "name": "sleep", "memoryMb": "4096"}
        ]
        health_ok = {"ok": True, "healthKind": "ok", "healthValue": "OK", "error": None}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(
                    GATE.adapter_runtime,
                    "processes_on",
                    return_value=(admission_procs, 1),
                ), \
                patch.object(
                    GATE.adapter_runtime, "run_host_bash", side_effect=run_host_bash
                ), \
                patch.object(
                    GATE,
                    "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 1, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", return_value=health_ok), \
                patch.object(
                    GATE, "_probe_device_processes", side_effect=probe_processes
                ), \
                patch.object(GATE, "HEALTH_POLL_SECONDS", 0.05), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.05), \
                patch.object(GATE, "HEALTH_PID_RESCAN_SECONDS", 0.05), \
                self.in_source():
            values = self.values()
            values["TILEFOUNDRY_NPU_OCCUPANCY_POLICY"] = "shared_quiet"
            rc, obs = GATE.run_l3_promotion(values, 60.0)
        return rc, obs, probe_mems_seen

    def test_l3_promotion_shared_quiet_aborts_on_hbm_spike(self):
        rc, obs, probe_mems = self._run_l3_shared_quiet_with_monitor(
            probe_mem_sequence=[4096, 32768, 32768, 32768],
            workload_sleep="10",
        )
        self.assertEqual(rc, 75)
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["failureCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        self.assertEqual(obs["healthMonitoring"]["outcome"], "aborted")
        self.assertEqual(
            obs["healthMonitoring"]["abort"]["kind"], "external_occupancy"
        )
        self.assertEqual(obs["healthMonitoring"]["abort"]["occupancy"], "shared(1)")
        self.assertEqual(probe_mems, [4096, 32768, 32768])

    def test_l3_promotion_shared_quiet_passes_under_steady_quiet_load(self):
        rc, obs, probe_mems = self._run_l3_shared_quiet_with_monitor(
            probe_mem_sequence=[4096],
            workload_sleep="0.35",
        )
        self.assertEqual(rc, 0)
        self.assertTrue(obs["exitCodeZero"])
        self.assertTrue(obs["promotionEligible"])
        self.assertTrue(obs["occupancyRecorded"])
        self.assertEqual(obs["deviceWindow"]["occupancy"], "shared(1)")
        self.assertEqual(obs["deviceWindow"]["occupancyPolicy"], "shared_quiet")
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        self.assertEqual(obs["healthMonitoring"]["outcome"], "completed")
        self.assertNotIn("abort", obs["healthMonitoring"])
        self.assertGreaterEqual(len(probe_mems), 2)
        self.assertTrue(all(mem == 4096 for mem in probe_mems))

    def test_candidate_kernel_compile_and_idempotence_failures_classify_as_candidate(self):
        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        candidate_cases = (
            (
                "explicit_candidate_kernel",
                self.producer_failure(
                    kind="candidate",
                    category="exception",
                    stage="kernel",
                    role="candidate",
                    node=AR_TESTS[0],
                    case="1",
                    exception_type="RuntimeError",
                    detail="exceeded vf stack size (0) during bisheng compile",
                ),
            ),
            (
                "legacy_infra_marked_vf_stack_overflow",
                self.producer_failure(
                    kind="infrastructure",
                    category="exception",
                    stage="kernel",
                    role="candidate",
                    node=AR_TESTS[0],
                    case="1",
                    exception_type="RuntimeError",
                    detail="Ascend compilation failed: exceeded vf stack size (0)",
                ),
            ),
            (
                "l3_cache_idempotence_violation",
                self.producer_failure(
                    kind="infrastructure",
                    category="assertion",
                    stage="cache_idempotence",
                    node=AR_TESTS[0],
                    case=None,
                    exception_type="ClassifiedTestFailure",
                    detail="L3 cache idempotence violation (Delta|Cache|=1)",
                ),
            ),
        )
        for label, failure in candidate_cases:
            with self.subTest(case=label):
                self.evidence = self.failure_evidence([failure])
                self.junit_xml = first_failed  # type: ignore[method-assign]
                rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
                self.assertEqual(rc, 1)
                self.assertFalse(obs["promotionEligible"])
                self.assertEqual(obs["errorCategory"], "candidate")
                self.assertEqual(obs["failureCategory"], "candidate")

    def test_baseline_hardware_and_interrupted_kernel_failures_stay_non_candidate(self):
        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        baseline_fail = self.producer_failure(
            kind="infrastructure",
            category="exception",
            stage="kernel",
            role="baseline",
            node=AR_TESTS[0],
            case="1",
            exception_type="RuntimeError",
            detail="exceeded vf stack size (0)",
        )
        self.evidence = self.failure_evidence([baseline_fail])
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertEqual(obs["failureCategory"], "environment")

        hw_fail = self.producer_failure(
            kind="infrastructure",
            category="exception",
            stage="kernel",
            role="candidate",
            node=AR_TESTS[0],
            case="1",
            exception_type="RuntimeError",
            detail="aclrtGetDevice failed: device offline E39999",
        )
        self.evidence = self.failure_evidence([hw_fail])
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertEqual(obs["failureCategory"], "environment")

        kernel_fail = self.producer_failure(
            kind="candidate",
            category="exception",
            stage="kernel",
            role="candidate",
            node=AR_TESTS[0],
            case="1",
            exception_type="RuntimeError",
            detail="exceeded vf stack size (0)",
        )
        base_interrupted = self.failure_evidence([kernel_fail])

        def interrupted_evidence(env):
            packet = base_interrupted(env)
            packet["interruption"]["errorCategory"] = "signal"
            return packet

        self.evidence = interrupted_evidence  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertEqual(obs["failureCategory"], "environment")

        # Bare JUnit failure without producer marker must still remain "infra".
        self.evidence = _Fixture.evidence.__get__(self, _Fixture)
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertEqual(obs["failureCategory"], "infra")

    def test_l3_numeric_failure_at_case3_not_preempted_by_downstream_missing_cases(self):
        required_cases = [str(i) for i in range(16)]
        case_states = {"0": "passed", "1": "passed", "2": "passed", "3": "failed"}
        candidate_fail = self.candidate_failure(node=AR_TESTS[0], case="3")
        collateral_missing = self.producer_failure(
            kind="coverage",
            category="missing_case",
            stage="coverage",
            node=AR_TESTS[0],
            case="4",
            detail=f"{AR_TESTS[0]}:4 was not completed",
        )
        base_builder = self.failure_evidence(
            [candidate_fail, collateral_missing],
            node_states={AR_TESTS[0]: case_states},
        )

        def build_evidence(env):
            packet = base_builder(env)
            packet["nodes"][0]["status"] = "failed"
            packet["nodes"][0]["requiredCases"] = list(required_cases)
            packet["coverage"]["qualified"] = False
            packet["coverage"]["cases"] = {
                "expected": 16,
                "completed": 4,
                "missing": [f"{AR_TESTS[0]}:{i}" for i in range(4, 16)],
                "unexpected": [],
            }
            return packet

        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = build_evidence  # type: ignore[method-assign]
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "numeric")
        self.assertEqual(obs["failureCategory"], "numeric")
        self.assertEqual(obs["evidence"]["timing"]["startedAt"], "2026-01-01T00:00:00Z")
        self.assertEqual(obs["evidence"]["timing"]["finishedAt"], "2026-01-01T00:01:00Z")

        # Direct _l3_classify / _l3_evidence_summary check.
        ev_packet = build_evidence({})
        junit = GATE._l3_junit_from_xml(first_failed()) if hasattr(GATE, "_l3_junit_from_xml") else None
        if junit is None:
            xml_path = self.root / "direct_junit.xml"
            xml_path.write_text(first_failed(), encoding="utf-8")
            junit = GATE._l3_junit(xml_path)
        category, coverage_complete, eligible, _notes = GATE._l3_classify(
            rc=1,
            expected=AR_TESTS,
            junit=junit,
            evidence=ev_packet,
            task="level4/attention_residual",
            operator="attention_residual",
            candidate=self.candidate.resolve(),
            candidate_digest=hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
            python=self.python.resolve(),
        )
        self.assertEqual(category, "numeric")
        self.assertFalse(coverage_complete)
        self.assertFalse(eligible)
        summary = GATE._l3_evidence_summary(
            ev_packet,
            expected_module=self.candidate.resolve(),
            candidate_relative="examples/ascend950pr_ops/attention_residual/kernels.py",
            candidate_digest=hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
        )
        assert summary is not None
        self.assertEqual(summary["timing"], ev_packet["timing"])

    def test_l3_missing_case_on_separate_unfailed_node_still_blocks_as_coverage(self):
        candidate_fail = self.candidate_failure(node=AR_TESTS[0], case="3")
        collateral_missing = self.producer_failure(
            kind="coverage",
            category="missing_case",
            stage="coverage",
            node=AR_TESTS[0],
            case="4",
            detail=f"{AR_TESTS[0]}:4 was not completed",
        )
        separate_missing = self.producer_failure(
            kind="coverage",
            category="missing_case",
            stage="coverage",
            node=AR_TESTS[1],
            case="15",
            detail=f"{AR_TESTS[1]}:15 was not completed",
        )
        base_builder = self.failure_evidence(
            [candidate_fail, collateral_missing, separate_missing],
            node_states={
                AR_TESTS[0]: {"0": "passed", "1": "passed", "2": "passed", "3": "failed"},
                AR_TESTS[1]: {"1": "passed"},
            },
        )

        junit_original = self.junit_xml

        def first_failed(**kwargs):
            return junit_original(statuses={
                name: ("failed" if name == AR_TESTS[0] else "passed")
                for name in AR_TESTS
            })

        self.evidence = base_builder  # type: ignore[method-assign]
        self.junit_xml = first_failed  # type: ignore[method-assign]
        rc, obs = self.run_l3(self.fake_runner(workload_rc=1))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["promotionEligible"])
        self.assertEqual(obs["errorCategory"], "coverage")
        self.assertEqual(obs["failureCategory"], "coverage")


class L4IdlePolicyCase(_Fixture, unittest.TestCase):
    def test_prep_blocks_before_inspection_on_a_shared_device(self):
        with patch.dict(os.environ, self.environment(), clear=True):
            values = self.values()
        inspect = patch.object(GATE, "_inspect_conda_cannbench_env")
        with patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(
                    GATE.adapter_runtime, "processes_on", return_value=([7], 1)
                ), \
                inspect as mocked, \
                self.in_source():
            rc, obs = GATE.run_cannbench_prep(values, 60.0)
        self.assertEqual(rc, 75)
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertTrue(obs["occupancyRecorded"])
        mocked.assert_not_called()

    def test_official_run_blocks_before_inspection_on_a_shared_device(self):
        with patch.dict(os.environ, self.environment(), clear=True):
            values = self.values()
        inspect = patch.object(GATE, "_inspect_conda_cannbench_env")
        with patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(
                    GATE.adapter_runtime, "processes_on", return_value=([7], 1)
                ), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root), \
                inspect as mocked, \
                self.in_source():
            rc, obs = GATE.run_cannbench(values, 60.0)
        self.assertEqual(rc, 75)
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertTrue(obs["occupancyRecorded"])
        mocked.assert_not_called()

    def test_official_timeout_keeps_null_counts_and_failed_phase(self):
        with patch.dict(os.environ, self.environment(), clear=True):
            values = self.values()
        info = {
            "missing": [],
            "pythonVersion": "3.12.16",
            "torchVersion": "2.13.0+cpu",
            "torchNpuVersion": "2.13.0",
            "torchCmakeDir": "/tmp/Torch",
            "torchNpuDir": "/tmp/torch_npu",
        }
        _, _, utils_dir = GATE._abi_slot(
            values, self.src, self.repo, self.work_root, info
        )
        (utils_dir / "cann_bench_utils").mkdir(parents=True)
        (utils_dir / "cann_bench_utils" / "_C.so").write_bytes(b"ELF")

        def timeout_runner(cmd, *, cwd, env, **kwargs):
            raise subprocess.TimeoutExpired(cmd="runner", timeout=1.0)

        with patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                patch.object(
                    GATE.adapter_runtime, "processes_on", return_value=([], 1)
                ), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root), \
                patch.object(
                    GATE.adapter_runtime, "run_host_bash", side_effect=timeout_runner
                ), \
                self.in_source():
            rc, obs = GATE.run_cannbench(values, 60.0)
        self.assertEqual(rc, 1)
        self.assertEqual(obs["errorCategory"], "deadline")
        self.assertIsNone(obs["caseCount"])
        self.assertIsNone(obs["passedCases"])
        self.assertIsNone(obs["timedCases"])
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        phases = {phase["name"]: phase for phase in obs["phases"]}
        self.assertEqual(phases["workload"]["status"], "failed")


class ManifestCase(unittest.TestCase):
    def test_l3_recipe_is_pinned_dynamic_and_budgeted(self):
        from agent_gates.manifest import Manifest

        manifest = Manifest(ADAPTER / "manifest.json")
        recipe = manifest.resolve_recipe("tilefoundry-npu.l3-promotion", manifest.root)
        self.assertTrue(recipe.uses_device)
        self.assertEqual(recipe.device_window, 60.0)
        self.assertEqual(recipe.timeout, 1920.0)
        self.assertEqual(recipe.config_check_timeout, 120.0)
        self.assertTrue(recipe.dynamic_target)
        # The control process must not import a project sitecustomize hook.
        # The pinned wrapper is invoked through bash so it can pass -S and -E
        # to Python before any project hook could load.
        self.assertEqual(recipe.argv[0], "bash")
        self.assertTrue(recipe.argv[2].endswith("/scripts/host_control.sh"))
        self.assertEqual(recipe.argv[3], "--l3-promotion")
        self.assertEqual(recipe.config_check[0], "bash")
        self.assertTrue(recipe.config_check[2].endswith("/scripts/host_control.sh"))
        self.assertEqual(recipe.config_check[3], "--check-config-l3")
        wrapper = (ADAPTER / "scripts/host_control.sh").read_text(encoding="utf-8")
        self.assertIn("-S", wrapper)
        self.assertIn("-E", wrapper)
        self.assertIn("tilefoundry_host_gate.py", wrapper)
        gate = manifest.gates["tilefoundry-npu-l3-promotion"]
        checks = gate["observationChecks"]
        for key in (
            "exitCodeZero",
            "promotionEligible",
            "coverageComplete",
            "cachePreflightOk",
            "windowOpened",
            "windowReleased",
            "occupancyRecorded",
            "candidateDigestRecorded",
        ):
            self.assertIs(checks[key], True)
        self.assertEqual(checks["testsFailed"], 0)
        for rel, expected in json.loads(
            (ADAPTER / "manifest.json").read_text(encoding="utf-8")
        )["files"].items():
            self.assertEqual(
                hashlib.sha256((ADAPTER / rel).read_bytes()).hexdigest(), expected
            )


if __name__ == "__main__":
    unittest.main()
