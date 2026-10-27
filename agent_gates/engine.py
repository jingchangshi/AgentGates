"""Run one adapter's gate DAG with source leases and independent evidence."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
from .evaluator import evaluate
from .evidence import EvidenceStore, aggregate
from .executor import execute, executable_identity, file_reference, now
from .locking import acquire
from .manifest import Manifest, ManifestError, SHA256
from .registry import Registry
from .source_identity import snapshot, source_root

ENV_KEYS = {"ascendnpu-ir": "ASCENDNPU_ENV_FILE", "tilelang-hivm": "TILELANG_HIVM_ENV_FILE",
            "tilelang-hivm-npu": "TILELANG_HIVM_ENV_FILE"}

# A configuration check explains itself on stderr. Only the last line is kept,
# bounded, and it is the Adapter's own message: no environment values, no
# other users' process ids, nothing the receipt should not carry.
REASON_LIMIT = 300

# Variables that change which toolchain, interpreter or libraries the recipe
# executes without appearing in the adapter's own prefix. They are inherited
# from the launching shell, so the receipt binds them by hash: two runs of the
# same recorded argv must not resolve to different compilers while their
# configuration hash stays identical. Values are never written to a receipt.
EXECUTION_ENV = ("PATH", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "HOME", "SHELL")


def config_reason(check):
    reference = check.get("stderr") or check.get("stdout") or {}
    path = reference.get("path")
    if not path:
        return ""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1][:REASON_LIMIT]


def environment(manifest: Manifest, env_file: Path | None = None, device_id: str | None = None):
    env = dict(os.environ)
    # An adapter reads only its own prefix. Inheriting the other adapter's
    # prefix would let a stray export (a build subdirectory, a Ninja job count)
    # steer this run without being part of what `configuration()` hashes.
    own = "ASCENDNPU_" if manifest.id == "ascendnpu-ir" else "TILELANG_HIVM_"
    for name in [name for name in env
                 if name.startswith(("ASCENDNPU_", "TILELANG_HIVM_")) and not name.startswith(own)]:
        del env[name]
    if env_file is not None:
        if not Path(env_file).is_absolute():
            raise ValueError("--env-file must be an absolute path")
        path = Path(env_file).resolve(strict=True)
        if not path.is_file():
            raise ValueError("--env-file must name a file")
        key = ENV_KEYS.get(manifest.id)
        if key is None:
            raise ValueError("--env-file unsupported for this adapter")
        env[key] = str(path)
    if device_id is not None:
        if not re.fullmatch(r"[0-9]+", device_id):
            raise ValueError("--device-id must be a nonnegative decimal ID")
        env["TILELANG_HIVM_DEVICE_ID"] = str(int(device_id))
    return env


def configuration(manifest, env):
    references = []
    key = ENV_KEYS.get(manifest.id)
    if key and env.get(key):
        path = Path(env[key])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"{key} must name an existing absolute file")
        references.append(file_reference(path.resolve()))
    elif key:
        directory_id = "tilelang-hivm" if manifest.id.startswith("tilelang-hivm") else manifest.id
        candidate = manifest.root.parent / directory_id / "env.sh"
        if candidate.is_file():
            references.append(file_reference(candidate.resolve()))
    # No secrets are copied to the receipt. Configuration files are hashed as data.
    prefix = "ASCENDNPU_" if manifest.id == "ascendnpu-ir" else "TILELANG_HIVM_"
    assignments = "\0".join(f"{key}={value}" for key, value in sorted(env.items()) if key.startswith(prefix))
    # Inherited variables that select a toolchain are bound by hash only. They
    # are recorded so two runs under different toolchains cannot present the
    # same configuration identity; `verify` deliberately does not re-derive
    # them, because the verifying shell's PATH is not the running shell's and
    # a changed PATH would otherwise mark every receipt STALE. No value is
    # stored: only the digest.
    inherited = "\0".join(f"{name}={env.get(name, '')}" for name in EXECUTION_ENV)
    return {"files": references,
            "environmentSha256": hashlib.sha256(assignments.encode()).hexdigest(),
            "inheritedEnvironmentSha256": hashlib.sha256(inherited.encode()).hexdigest()}


def artifact_inputs(source, manifest, env):
    """Known existing compiler products, before an observation-only regression."""
    paths = []
    if manifest.id.startswith("tilelang-hivm"):
        paths.append(source / "tilelang/__init__.py")
        # Native search paths may point outside the default build directory.
        # Snapshot all shared products under this checkout before testing.
        for directory, dirs, files in os.walk(source, followlinks=False):
            dirs[:] = [name for name in dirs if name != ".git"]
            paths.extend(Path(directory) / name for name in files if ".so" in name and (name.endswith(".so") or ".so." in name))
    elif manifest.id == "ascendnpu-ir":
        build = source / env.get("ASCENDNPU_BUILD_SUBDIR", "build")
        paths.extend(build / "bin" / name for name in ("bishengir-opt", "bishengir-compile"))
    result = {}
    for path in paths:
        if path.is_file():
            resolved = path.resolve()
            result[str(resolved)] = file_reference(resolved)
    return result


def collect_artifacts(observed, source):
    artifacts = observed.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("artifacts observation must be a list")
    references = []
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or not SHA256.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError("invalid artifact identity observation")
        path = Path(artifact["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("artifact must name an existing absolute host file")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source):
            raise ValueError("artifact escapes source/build root")
        reference = file_reference(resolved)
        if reference["sha256"] != artifact["sha256"]:
            raise ValueError("observed artifact hash differs from host bytes")
        if str(resolved) not in seen:
            references.append(reference)
            seen.add(str(resolved))
    return references


class Engine:
    def __init__(self, *, registry=None, store=None, lock_root=None):
        self.registry = registry or Registry()
        self.store = store or EvidenceStore()
        self.lock_root = lock_root

    def doctor(self, adapter, source, *, env_file=None, device_id=None):
        manifest = self.registry.get(adapter)
        source = source_root(Path(source))
        if self.store.root.resolve().is_relative_to(source):
            raise ValueError("evidence directory must be outside the source root")
        env = environment(manifest, env_file, device_id)
        run_id, directory = self.store.create_run()
        checks, failures = [], []
        for recipe_id in manifest.recipes:
            recipe = manifest.resolve_recipe(recipe_id, source)
            try:
                manifest.validate_files()
                executable_identity(recipe.argv, recipe.cwd, env)
                if not recipe.config_check:
                    failures.append(f"CONFIG_CHECK_MISSING: {recipe_id}")
                    continue
                result = execute(recipe.config_check, cwd=recipe.cwd, env=env, timeout=min(120, recipe.timeout),
                                 log_dir=directory / "doctor" / recipe_id)
                checks.append({"recipeId": recipe_id, **result})
                if result["exitCode"] != 0 or result["timedOut"] or result["interrupted"]:
                    failures.append(f"CONFIG_CHECK_FAILED: {recipe_id}; inspect {result['stderr']['path']}")
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(str(exc))
        return {"runId": run_id, "adapter": adapter, "source": str(source), "status": "BLOCKED_ENV" if failures else "PASS",
                "checks": checks, "failures": failures, "logDirectory": str(directory)}

    def run(self, adapter, source, selected, *, env_file=None, device_id=None):
        manifest = self.registry.get(adapter)
        source = source_root(Path(source))
        if self.store.root.resolve().is_relative_to(source):
            raise ValueError("evidence directory must be outside the source root")
        env = environment(manifest, env_file, device_id)
        order = manifest.order(selected)
        run_id, directory = self.store.create_run()
        receipt = {"schemaVersion": 1, "runId": run_id, "startedAt": now(), "finishedAt": None,
                   "adapter": {"id": manifest.id, "path": str(manifest.path), "sha256": manifest.manifest_hash,
                               "files": manifest.files}, "selectedGates": selected, "gates": [],
                   "source": snapshot(source), "configuration": {}, "status": "NOT_RUN",
                   "trust": "local-user-integrity", "historicalPassReuse": False}
        try:
            receipt["configuration"] = configuration(manifest, env)
            resources = ["source:" + str(source)]
            build_prefix = "ASCENDNPU" if manifest.id == "ascendnpu-ir" else "TILELANG_HIVM"
            build = (source / env.get(build_prefix + "_BUILD_SUBDIR", "build")).resolve()
            resources.append("build:" + str(build))
            if manifest.id == "tilelang-hivm-npu":
                device = env.get("TILELANG_HIVM_DEVICE_ID", "")
                if not re.fullmatch(r"[0-9]+", device):
                    raise ValueError("explicit TILELANG_HIVM_DEVICE_ID or --device-id authorization required")
                resources.append("npu:" + str(int(device)))
                receipt["authorizedDeviceId"] = str(int(device))
            with acquire(resources, root=self.lock_root) as lease_fds:
                # Capture after acquiring locks; another AgentGates run may just have finished.
                receipt["source"] = snapshot(source)
                for gate_id in order:
                    definition = manifest.gates[gate_id]
                    recipe = manifest.resolve_recipe(definition["recipeId"], source)
                    gate = {"gateId": gate_id, "recipeId": recipe.id, "checks": definition["observationChecks"],
                            "status": "NOT_RUN", "failures": [], "observations": {}, "artifacts": []}
                    receipt["gates"].append(gate)
                    prior = {item["gateId"]: item["status"] for item in receipt["gates"][:-1]}
                    if any(prior.get(dependency) != "PASS" for dependency in definition.get("dependsOn", [])):
                        gate["failures"] = ["DEPENDENCY_NOT_PASSED"]
                        continue
                    try:
                        self._gate(manifest, source, recipe, gate, receipt, env, directory, lease_fds)
                    except KeyboardInterrupt:
                        gate.update(status="FAIL", failures=["INTERRUPTED"])
                        break
                    except (OSError, ValueError, RuntimeError) as exc:
                        gate.update(status="INVALID_EVIDENCE" if isinstance(exc, ManifestError) else "BLOCKED_ENV", failures=[str(exc)])
                # A later gate may have rebuilt an earlier gate's products.
                # A final PASS must bind every gate to the final artifacts.
                for gate in receipt["gates"]:
                    if gate["status"] == "PASS" and any(not self.store._matches(item) for item in gate["artifacts"]):
                        gate.update(status="STALE", failures=["ARTIFACT_IDENTITY_CHANGED"])
                current = snapshot(source)
                if current != receipt["source"]:
                    for gate in receipt["gates"]:
                        if gate["status"] == "PASS":
                            gate.update(status="STALE", failures=["SOURCE_IDENTITY_CHANGED"])
                manifest.validate_files()
                if configuration(manifest, env) != receipt["configuration"]:
                    for gate in receipt["gates"]:
                        if gate["status"] == "PASS":
                            gate.update(status="STALE", failures=["CONFIGURATION_IDENTITY_CHANGED"])
        except (OSError, ValueError, RuntimeError) as exc:
            existing = {item["gateId"] for item in receipt["gates"]}
            for gate_id in order:
                if gate_id not in existing:
                    definition = manifest.gates[gate_id]
                    receipt["gates"].append({"gateId": gate_id, "recipeId": definition["recipeId"],
                                             "checks": definition["observationChecks"], "status": "BLOCKED_ENV",
                                             "failures": [str(exc)], "observations": {}, "artifacts": []})
            for gate in receipt["gates"]:
                if gate["status"] == "PASS":
                    gate.update(status="INVALID_EVIDENCE", failures=[str(exc)])
        existing = {item["gateId"] for item in receipt["gates"]}
        for gate_id in order:
            if gate_id not in existing:
                definition = manifest.gates[gate_id]
                receipt["gates"].append({"gateId": gate_id, "recipeId": definition["recipeId"],
                                         "checks": definition["observationChecks"], "status": "NOT_RUN",
                                         "failures": ["RUN_INTERRUPTED"], "observations": {}, "artifacts": []})
        receipt["status"] = aggregate(receipt["gates"])
        receipt["finishedAt"] = now()
        try:
            path = self.store.publish(receipt)
        except (OSError, ValueError) as exc:
            # Everything before this point is already on disk: the logs and
            # process records live in the run directory even when the signed
            # receipt cannot be written. Name that directory so the run is not
            # lost silently, and keep the failure a RuntimeError like every
            # other evidence-path fault rather than letting a bare ValueError
            # escape.
            raise RuntimeError(f"receipt could not be published ({exc}); "
                               f"run evidence retained in {directory}") from exc
        return {**receipt, "receiptPath": str(path)}

    def _gate(self, manifest, source, recipe, gate, receipt, env, directory, lease_fds):
        manifest.validate_files()
        gate["sourceBefore"] = snapshot(source)
        if gate["sourceBefore"] != receipt["source"]:
            gate.update(status="STALE", failures=["SOURCE_IDENTITY_CHANGED"])
            return
        gate["executable"] = executable_identity(recipe.argv, recipe.cwd, env)
        if not recipe.config_check and manifest.id in ENV_KEYS:
            gate.update(status="BLOCKED_ENV", failures=["CONFIG_CHECK_MISSING"])
            return
        if recipe.config_check:
            check = execute(recipe.config_check, cwd=recipe.cwd, env=env, timeout=min(120, recipe.timeout),
                            log_dir=directory / gate["gateId"] / "config", lease_fds=lease_fds)
            gate["configCheck"] = check
            if check["exitCode"] != 0 or check["timedOut"] or check["interrupted"]:
                # The Adapter's own refusal is the actionable reason (an
                # unconfigured machine, a card another job holds). Keep one
                # bounded line on the Gate so a reader does not have to open
                # the log to learn why the Gate never started. The failure
                # marker itself stays machine-readable and unparameterised.
                gate.update(status="BLOCKED_ENV", failures=["CONFIG_CHECK_FAILED"],
                            configReason=config_reason(check))
                return
        manifest.validate_files()
        if snapshot(source) != gate["sourceBefore"]:
            gate.update(status="STALE", failures=["SOURCE_CHANGED_DURING_CONFIG_CHECK"])
            return
        if configuration(manifest, env) != receipt["configuration"]:
            gate.update(status="STALE", failures=["CONFIGURATION_IDENTITY_CHANGED"])
            return
        inputs = artifact_inputs(source, manifest, env)
        gate["artifactInputs"] = list(inputs.values())
        process = execute(recipe.argv, cwd=recipe.cwd, env=env, timeout=recipe.timeout,
                          log_dir=directory / gate["gateId"] / "recipe", lease_fds=lease_fds)
        gate["process"] = process
        gate.update(evaluate(process, gate["checks"]))
        if type(process.get("exitCode")) is int and process["exitCode"] in recipe.blocked_exit_codes \
                and not process.get("timedOut") and not process.get("interrupted"):
            # The Adapter uses a dedicated exit code for "the environment is not
            # available to this Gate" - a card another job holds, for instance.
            # Nothing was dispatched and no assertion ran, so recording FAIL
            # would claim a test failed that never started. This is the same
            # classification the configuration check produces for the same
            # condition, and it is still not a PASS: `aggregate` keeps
            # BLOCKED_ENV out of a passing run. The evaluation's observation
            # failures are dropped on purpose - they only say "the report was
            # empty", which is a consequence of not running, not a finding.
            # A timeout or an interrupt is a different event even when the
            # supervisor's final status happens to match: the child may have
            # been killed mid-run after starting work, so it stays a failure.
            gate.update(status="BLOCKED_ENV", failures=[f"BLOCKED_EXIT_CODE: {process['exitCode']}"],
                        configReason=config_reason(process))
        gate["sourceAfter"] = snapshot(source)
        if gate["sourceAfter"] != gate["sourceBefore"]:
            gate.update(status="STALE", failures=gate["failures"] + ["SOURCE_IDENTITY_CHANGED"])
        manifest.validate_files()
        try:
            gate["artifacts"] = collect_artifacts(gate["observations"], source)
            # Exit-code build gates still bind the actual products they created.
            if not gate["artifacts"]:
                gate["artifacts"] = list(artifact_inputs(source, manifest, env).values())
            if "regression" in recipe.id or manifest.id.endswith("-npu") or "loadedNativeSha256MatchesHost" in gate["checks"]:
                for artifact in gate["artifacts"]:
                    if artifact["path"] in inputs and artifact != inputs[artifact["path"]]:
                        gate.update(status="STALE", failures=gate["failures"] + ["ARTIFACT_CHANGED_DURING_TEST"])
                    elif ".so" in Path(artifact["path"]).name and artifact["path"] not in inputs:
                        gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + ["ARTIFACT_PRETEST_IDENTITY_MISSING"])
            if any(key in gate["checks"] for key in ("loadedNativeSha256MatchesHost", "importedModuleSha256MatchesHost")) and not gate["artifacts"]:
                gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + ["ARTIFACT_IDENTITIES_MISSING"])
        except (OSError, ValueError) as exc:
            gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + [str(exc)])
