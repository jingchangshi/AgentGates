"""Run one adapter's gate DAG with source leases and independent evidence."""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import re
import sys
from . import dynamic_target
from .evaluator import evaluate
from .docker_quarantine import is_quarantined_recipe, quarantine_reason
from .evidence import EvidenceStore, aggregate
from .executor import execute, executable_identity, file_reference, now
from . import machine_env
from .locking import acquire, device_lock_root as shared_device_lock_root, lock_path
from .manifest import Manifest, ManifestError, SHA256, strict_json
from .registry import Registry
from .source_identity import snapshot, source_root

ENV_KEYS = {"ascendnpu-ir": "ASCENDNPU_ENV_FILE", "tilelang-hivm": "TILELANG_HIVM_ENV_FILE",
            "tilelang-hivm-npu": "TILELANG_HIVM_ENV_FILE", "tla-npu": "TLA_NPU_ENV_FILE",
            "forgekernel-ascend-npu": "FK_ASCEND_ENV_FILE", "bska-npu": "BSKA_NPU_ENV_FILE",
            "tilelang-npu": "TILELANG_NPU_ENV_FILE",
            "tilefoundry-npu": "TILEFOUNDRY_NPU_ENV_FILE"}

# The environment prefix each Adapter owns. An Adapter reads only its own prefix,
# so a stray export for one Adapter cannot steer another; this map is the single
# place that association is written.
OWN_PREFIX = {"ascendnpu-ir": "ASCENDNPU_", "tilelang-hivm": "TILELANG_HIVM_",
              "tilelang-hivm-npu": "TILELANG_HIVM_", "tla-npu": "TLA_NPU_",
              "forgekernel-ascend-npu": "FK_ASCEND_", "bska-npu": "BSKA_NPU_",
              "tilelang-npu": "TILELANG_NPU_",
              "tilefoundry-npu": "TILEFOUNDRY_NPU_"}

# The Adapters that require an explicitly authorized NPU device. Each names its
# own device variable and leases an npu:<id> on it, so two AgentGates runs cannot
# drive one card at the same time. The lease coordinates OUR runs with each
# other; it does not, and cannot, exclude other tenants of a shared machine.
DEVICE_ADAPTERS = {"tilelang-hivm-npu": "TILELANG_HIVM_DEVICE_ID",
                   "tilelang-npu": "TILELANG_NPU_DEVICE_ID",
                   "tla-npu": "TLA_NPU_DEVICE_ID",
                   "forgekernel-ascend-npu": "FK_ASCEND_DEVICE_ID",
                   "bska-npu": "BSKA_NPU_DEVICE_ID",
                   "tilefoundry-npu": "TILEFOUNDRY_NPU_DEVICE_ID"}

# The variables a windowed Recipe reads to open the device lease itself. The
# engine writes them only for a recipe that declares `deviceWindow`; every other
# recipe keeps the whole-recipe lease it always had, so a manifest that says
# nothing about windows behaves exactly as before.
#
# The path is passed rather than derived by the Adapter: the lease file is a
# hash of the resource name, and a second implementation of that hash is how two
# different locks get taken on one card. The recipe flocks this exact file.
DEVICE_LEASE_PATH = "AGENT_GATES_DEVICE_LEASE_PATH"
DEVICE_LEASE_RESOURCE = "AGENT_GATES_DEVICE_LEASE_RESOURCE"
DEVICE_LEASE_WAIT = "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS"
LEASE_ENVIRONMENT = (DEVICE_LEASE_PATH, DEVICE_LEASE_RESOURCE, DEVICE_LEASE_WAIT)

# The Adapter that reads each Adapter variable for the exclusivity policy. The
# engine forwards the operator choice, but the variable it writes is the one the
# receiving Adapter actually parses; writing another Adapter prefix here would
# look forwarded and be silently ignored by the Gate.
EXCLUSIVE_VARIABLE = {"tilelang-hivm-npu": "TILELANG_HIVM_REQUIRE_EXCLUSIVE_DEVICE",
                      "tilelang-npu": "TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE",
                      "tla-npu": "TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE",
                      "forgekernel-ascend-npu": "FK_ASCEND_REQUIRE_EXCLUSIVE_DEVICE",
                      "bska-npu": "BSKA_NPU_REQUIRE_EXCLUSIVE_DEVICE",
                      "tilefoundry-npu": "TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE"}

# A configuration check explains itself on stderr. Only the last line is kept,
# bounded, and it is the Adapter's own message: no environment values, no
# other users' process ids, nothing the receipt should not carry.
REASON_LIMIT = 300

# A configuration check may explain a structured refusal on one stderr line. The
# engine keeps only the last such line, validates its tiny grammar, and records
# it beside the human-readable reason. A malformed line is ignored: diagnostics
# must never become a second, unreviewed classification channel.
CONFIG_OBSERVATION_PREFIX = "AGENT_GATES_CONFIG:"
CONFIG_CATEGORIES = frozenset({"CONFIG", "DEPENDENCY", "DEVICE", "RESOURCE",
                               "TARGET", "CACHE"})

# Variables that change which toolchain, interpreter or libraries the recipe
# executes without appearing in the adapter's own prefix. They are inherited
# from the launching shell, so the receipt binds them by hash: two runs of the
# same recorded argv must not resolve to different compilers while their
# configuration hash stays identical. Values are never written to a receipt.
EXECUTION_ENV = ("PATH", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH",
                 "HOME", "SHELL", "AGENT_GATES_DEVICE_LOCK_ROOT")


def _tail_text(reference, limit=65536):
    """The last bounded bytes of one log, decoded, without reading a whole file.

    A misbehaving configuration check can write an unbounded log before it is
    killed; diagnostics read a fixed tail so recording a reason cannot become a
    second resource problem.
    """
    path = reference.get("path")
    if not path:
        return ""
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read()
    except OSError:
        return ""
    if size > limit:
        # The first line is likely cut mid-way; a diagnostic line must be whole.
        data = data.split(b"\n", 1)[-1]
    return data.decode("utf-8", "replace")


def config_reason(check):
    # The human line usually lands on stderr, but a check that puts its
    # observation there and its prose on stdout must not lose the prose.
    for name in ("stderr", "stdout"):
        text = _tail_text(check.get(name) or {})
        lines = [line.strip() for line in text.splitlines() if line.strip()
                 and not line.lstrip().startswith(CONFIG_OBSERVATION_PREFIX)]
        if lines:
            return lines[-1][:REASON_LIMIT]
    return ""


def config_observation(check):
    """The last well-formed structured configuration observation, or None."""
    for name in ("stderr", "stdout"):
        text = _tail_text(check.get(name) or {})
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith(CONFIG_OBSERVATION_PREFIX):
                continue
            payload = line[len(CONFIG_OBSERVATION_PREFIX):].strip()
            try:
                value = strict_json(payload)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict) or set(value) != {"category", "detail"}:
                continue
            category, detail = value["category"], value["detail"]
            if not isinstance(category, str) or category not in CONFIG_CATEGORIES:
                continue
            if not isinstance(detail, str):
                continue
            detail = " ".join(detail.split())[:REASON_LIMIT]
            if not detail:
                continue
            return {"category": category, "detail": detail}
    return None


def config_check_timeout(recipe):
    """The independent configuration-check bound, or the historical default."""
    if recipe.config_check_timeout is not None:
        return recipe.config_check_timeout
    return min(120, recipe.timeout)


def candidate_matches(source, request):
    """Whether the bound candidate still has the bytes the receipt records."""
    if not isinstance(request, dict):
        return False
    candidate = request.get("candidate")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("relativePath"), str):
        return False
    try:
        return dynamic_target.candidate_reference(source, candidate["relativePath"]) == candidate
    except (OSError, ValueError):
        return False


def environment(manifest: Manifest, env_file: Path | None = None, device_id: str | None = None,
                require_exclusive_device: bool = False):
    env = dict(os.environ)
    # An adapter reads only its own prefix. Inheriting the other adapter's
    # prefix would let a stray export (a build subdirectory, a Ninja job count)
    # steer this run without being part of what `configuration()` hashes.
    own = OWN_PREFIX.get(manifest.id, "")
    for name in [name for name in env
                 if name.startswith(("ASCENDNPU_", "TILELANG_HIVM_", "TILELANG_NPU_", "TLA_NPU_",
                                     "FK_ASCEND_", "BSKA_NPU_", "TILEFOUNDRY_NPU_"))
                 and not name.startswith(own)]:
        del env[name]
    py_bin = os.path.dirname(os.path.abspath(sys.executable))
    path_parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    if py_bin not in path_parts:
        env["PATH"] = os.pathsep.join([py_bin, *path_parts]) if path_parts else py_bin
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
        # Only a device Adapter has a variable to write. Naming it from the map
        # keeps one Adapter device from being handed to another.
        variable = DEVICE_ADAPTERS.get(manifest.id)
        if variable is not None:
            env[variable] = str(int(device_id))
    # Exclusivity is a policy the operator chose in the local configuration, not
    # a precondition for running. Forwarding it lets the Adapter decide whether a
    # shared card is acceptable, which is the only layer that can see occupancy.
    # A lease handle is handed to one specific recipe for one specific window.
    # Inheriting it would let a stale export point a recipe at a lease file the
    # engine never took, so every run starts from "no lease was published".
    # The target/candidate pair follows the same rule: it is published only to a
    # recipe that declares it, never inherited from the launching shell.
    for name in (*LEASE_ENVIRONMENT, *dynamic_target.REQUEST_ENVIRONMENT):
        env.pop(name, None)
    exclusive_variable = EXCLUSIVE_VARIABLE.get(manifest.id)
    if exclusive_variable is not None:
        if require_exclusive_device:
            env[exclusive_variable] = "1"
        else:
            env.pop(exclusive_variable, None)
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
    prefix = OWN_PREFIX.get(manifest.id, manifest.id.upper().replace("-", "_") + "_")
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


def build_directory(source: Path, manifest: Manifest, env) -> Path:
    """The build tree a run will write, resolved from the Adapter's machine file.

    The build directory is a machine fact that lives in the same `env.sh` the
    Adapter reads. Resolving it here from the same file is what keeps the lease
    consistent with the work: reading `ASCENDNPU_BUILD_SUBDIR` from the launching
    shell while the Adapter read the file let one run hold a lock on `build`
    while building `build-fast`, so two runs could write one tree.

    The Adapter's own file wins, and an inherited export is used only when no
    file defines the key - the same rule the Adapter applies, so the two cannot
    disagree about which directory is leased.
    """
    if manifest.id not in machine_env.ADAPTERS:
        # A fixture or third-party adapter keeps its own convention: it was never
        # migrated to a machine file, so the launching environment is all it has.
        return (source / env.get(manifest.id.upper().replace("-", "_") + "_BUILD_SUBDIR",
                                 "build")).resolve()
    try:
        machine = machine_env.read(manifest.id, env, env_file=None)
        name = machine.build_subdir
    except machine_env.MachineEnvError as exc:
        raise ValueError(str(exc)) from exc
    return (source / name).resolve()


def artifact_inputs(source, manifest, env):
    """Known existing compiler products, before an observation-only regression."""
    paths = []
    if manifest.id.startswith("tilelang-hivm") or manifest.id == "tilelang-npu":
        paths.append(source / "tilelang/__init__.py")
        # Native search paths may point outside the default build directory.
        # Snapshot all shared products under this checkout before testing.
        def _refuse(error):
            # A shared product this walk cannot see is one whose identity cannot
            # be bound, so refuse rather than silently narrow the snapshot.
            raise error

        for directory, dirs, files in os.walk(source, followlinks=False,
                                            onerror=_refuse):
            dirs[:] = [name for name in dirs if name != ".git"]
            paths.extend(Path(directory) / name for name in files if ".so" in name and (name.endswith(".so") or ".so." in name))
    elif manifest.id == "ascendnpu-ir":
        build = build_directory(source, manifest, env)
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


def is_observation_only_recipe(manifest_id: str, recipe_id: str, checks) -> bool:
    """True for test/regression recipes whose pre-existing .so inputs must not change during execution."""
    step = recipe_id.rpartition(".")[2] or recipe_id
    if step == "build" or step.endswith("-build"):
        return False
    return (
        "regression" in recipe_id
        or manifest_id.endswith("-npu")
        or "loadedNativeSha256MatchesHost" in checks
    )


def device_lease_plan(manifest, order, source):
    """How this run holds its device card: whole-recipe, per-window, or not at all.

    Returns ``(variable, windows)``. ``variable`` is the Adapter's device
    variable (None for an Adapter that never names a card), and ``windows`` maps
    a gate id to the recipe that opens the lease itself.

    A recipe that declares ``deviceWindow`` is handed the lease instead of
    having it held for it. Holding the whole-recipe lease *and* handing the same
    lease to a child would deadlock: the engine owns the flock, the child waits
    for it, and no window ever opens. So the two modes are mutually exclusive in
    one run, and a manifest that mixes them is refused as a manifest error
    rather than silently running one of its device gates unprotected.
    """
    variable = DEVICE_ADAPTERS.get(manifest.id)
    if variable is None:
        return None, {}
    # Only a selected gate that actually drives the card needs a lease. A
    # device Adapter may also ship host-only steps - installing a build, reading
    # a configuration - and demanding a device allocation for those would make
    # work that never touches a card impossible to run on a busy machine.
    windows, whole = {}, []
    for gate_id in order:
        recipe = manifest.resolve_recipe(manifest.gates[gate_id]["recipeId"], source)
        if not recipe.uses_device:
            continue
        if recipe.device_window is None:
            whole.append(gate_id)
        else:
            windows[gate_id] = recipe
    if not windows and not whole:
        # Nothing selected touches the card: no id, no lease, no window.
        return None, {}
    if windows and whole:
        held, handed = ", ".join(sorted(whole)), ", ".join(sorted(windows))
        raise ManifestError(
            f"adapter declares both whole-recipe and windowed device gates ({held} vs {handed}); "
            "one run cannot hold the card and also hand the same lease to a recipe")
    return variable, windows


def device_window(observations, lease):
    """The window facts a windowed Recipe must publish, checked against the plan.

    The engine cannot see when the child held the flock, so the child states it
    and this function decides whether that statement is admissible. What is
    checked is that the child names the lease it was given - not one it chose -
    that both ends of the window exist in order, and that occupancy is a value
    this contract defines. A run that publishes nothing is refused: an unshown
    window is not a short window.

    Occupancy is recorded, never gated: another tenant on the card is normal on a
    shared machine and must not turn into a failure.
    """
    reported = observations.get("deviceWindow")
    if not isinstance(reported, dict):
        return None, "DEVICE_WINDOW_MISSING"
    if reported.get("resource") != lease["resource"]:
        return None, "DEVICE_WINDOW_RESOURCE_MISMATCH"
    if reported.get("path") != lease["path"]:
        return None, "DEVICE_WINDOW_PATH_MISMATCH"
    started, released = reported.get("startedAt"), reported.get("releasedAt")
    if not isinstance(started, str) or not isinstance(released, str) or started > released:
        return None, "DEVICE_WINDOW_TIMES_INVALID"
    occupancy = reported.get("occupancy")
    if not isinstance(occupancy, str) or not (occupancy == "free" or occupancy == "unknown"
                                              or re.fullmatch(r"shared\([0-9]+\)", occupancy)):
        return None, "DEVICE_WINDOW_OCCUPANCY_INVALID"
    waited = reported.get("waited")
    if type(waited) not in (int, float) or waited < 0:
        return None, "DEVICE_WINDOW_WAIT_INVALID"
    window = {"resource": reported["resource"], "path": reported["path"],
              "startedAt": started, "releasedAt": released, "occupancy": occupancy,
              "waited": float(waited)}
    # Optional durations, carried through so a receipt can show that the window
    # was dominated by work rather than by setup. They are RECORDED, never gated:
    # a slow machine is not a failed test, and a bound on elapsed time would make
    # the verdict depend on what else the host was doing. A value of the wrong
    # type is dropped rather than trusted.
    for name in ("heldSeconds", "probeSeconds", "warmupSeconds"):
        value = reported.get(name)
        if type(value) in (int, float) and value >= 0:
            window[name] = float(value)
    return window, None


class Engine:
    def __init__(self, *, registry=None, store=None, lock_root=None, device_lock_root=None):
        self.registry = registry or Registry()
        self.store = store or EvidenceStore()
        self.lock_root = lock_root
        # Device leases are host-wide, not evidence-directory-wide. An explicit
        # value exists for tests and unusual deployments; production V1 and V2
        # both resolve the same canonical root (or its one environment override)
        # so two processes cannot drive one card at the same time.
        self.device_lock_root = device_lock_root

    def _device_root(self) -> Path:
        if self.device_lock_root is not None:
            return Path(self.device_lock_root)
        return shared_device_lock_root()

    def doctor(self, adapter, source, *, selected=None, env_file=None, device_id=None,
               require_exclusive_device=False, target=None, candidate=None):
        """Probe the configuration checks of one adapter, optionally for one route.

        Every route is the default, because a machine diagnostic that silently
        narrowed itself would answer a question nobody asked. A caller that names
        gates is instead naming the route it can actually run - a host-only
        machine has no Docker - so the selection is expanded with the same
        `order` helper `run` uses, and an unknown id is refused there rather
        than being diagnosed away.

        `selected` is deliberately not a required positional: no caller, and no
        receipt, changes meaning because this argument exists. An empty selection
        is the all-route default and never "check nothing", because a PASS over
        zero recipes would be a vacuous claim rather than a diagnosis.
        """
        manifest = self.registry.get(adapter)
        source = source_root(Path(source))
        if self.store.root.resolve().is_relative_to(source):
            raise ValueError("evidence directory must be outside the source root")
        env = environment(manifest, env_file, device_id, require_exclusive_device)
        # Expanded before any run directory exists, and in the same position
        # `run` expands its own selection, so a mistyped gate costs the caller
        # the same error with the same precedence rather than an evidence tree
        # to clean up.
        gates = manifest.order(selected) if selected else None
        request = None
        if gates is None:
            if target is not None or candidate is not None:
                raise dynamic_target.TargetRequestError(
                    "target and candidate require an explicit gate selection")
        else:
            # A scoped diagnosis speaks for the route that will actually run,
            # so a dynamic route must be probed with the same pair a run needs.
            request = dynamic_target.resolve_request(
                manifest, gates, source, target=target, candidate=candidate)
        run_id, directory = self.store.create_run()
        checks, failures, checked, deduplicated = [], [], [], {}
        for recipe_id in self._doctor_recipes(manifest, gates, source, deduplicated):
            recipe = manifest.resolve_recipe(recipe_id, source)
            check_env = dynamic_target.apply_environment(env, recipe, request)
            if is_quarantined_recipe(manifest.id, recipe_id, recipe.argv):
                msg = quarantine_reason(f"recipe {recipe_id!r}")
                checks.append({
                    "recipeId": recipe_id,
                    "exitCode": 75,
                    "timedOut": False,
                    "interrupted": False,
                    "reason": msg,
                })
                checked.append(recipe_id)
                failures.append(msg)
                continue
            try:
                manifest.validate_files()
                executable_identity(recipe.argv, recipe.cwd, env)
                if not recipe.config_check:
                    failures.append(f"CONFIG_CHECK_MISSING: {recipe_id}")
                    continue
                result = execute(recipe.config_check, cwd=recipe.cwd, env=check_env,
                                 timeout=config_check_timeout(recipe),
                                 log_dir=directory / "doctor" / recipe_id)
                checks.append({"recipeId": recipe_id, **result})
                checked.append(recipe_id)
                if result["exitCode"] != 0 or result["timedOut"] or result["interrupted"]:
                    failures.append(f"CONFIG_CHECK_FAILED: {recipe_id}; inspect {result['stderr']['path']}")
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(str(exc))
        # The scope is stated, not implied. A reader must be able to tell "every
        # route on this machine is configured" from "the route I named is
        # configured", and `allRoutes` is the single bit that says which one this
        # report is: false unless every route of the adapter was probed.
        return {"runId": run_id, "adapter": adapter, "source": str(source),
                "status": "BLOCKED_ENV" if failures else "PASS",
                "allRoutes": gates is None,
                "selectedGates": self._doctor_gates(manifest, gates),
                "checkedRecipes": checked, "deduplicatedRecipes": deduplicated,
                "checks": checks, "failures": failures, "logDirectory": str(directory)}

    @staticmethod
    def _doctor_gates(manifest, gates):
        """The gates this report speaks for: the closure, or every gate."""
        return list(gates) if gates is not None else manifest.order(list(manifest.gates))

    @staticmethod
    def _doctor_recipes(manifest, gates, source, deduplicated):
        """Which recipes to check, and which were folded into an identical one.

        Every selected recipe is checked once. Two routes may also share one
        configuration check - the Docker and host routes of one adapter can read
        the same machine file - and running byte-identical argv twice would spend
        a subprocess to learn the same fact. Folding is recorded in
        `deduplicated` so the report still accounts for every selected recipe:
        a folded recipe is checked THROUGH its twin, not silently dropped.
        """
        if gates is None:
            # Untouched legacy scope: every declared recipe, each checked once,
            # in manifest order. No folding, because that would change how many
            # checks an existing caller sees.
            return list(manifest.recipes)
        recipes, seen, owner = [], set(), {}
        for gate_id in gates:
            recipe_id = manifest.gates[gate_id]["recipeId"]
            if recipe_id in seen:
                continue
            seen.add(recipe_id)
            recipe = manifest.resolve_recipe(recipe_id, source)
            if not recipe.config_check:
                recipes.append(recipe_id)
                continue
            identity = (recipe.config_check, recipe.cwd)
            if identity in owner:
                deduplicated[recipe_id] = owner[identity]
                continue
            owner[identity] = recipe_id
            recipes.append(recipe_id)
        return recipes

    def run(self, adapter, source, selected, *, env_file=None, device_id=None,
            require_exclusive_device=False, target=None, candidate=None,
            expected_source: dict | None = None,
            expected_candidate: dict | None = None):
        """Execute one route and publish its signed receipt.

        A queued request (`submit` -> worker) passes the source and candidate
        identities frozen at admission as `expected_source` and
        `expected_candidate`. They are checked here, before a run directory
        exists and again under the locks, so a worker that starts after the tree
        changed refuses instead of executing bytes nobody submitted. V1 callers
        pass neither and keep the old behaviour.
        """
        manifest = self.registry.get(adapter)
        source = source_root(Path(source))
        if self.store.root.resolve().is_relative_to(source):
            raise ValueError("evidence directory must be outside the source root")
        env = environment(manifest, env_file, device_id, require_exclusive_device)
        order = manifest.order(selected)
        if expected_candidate is not None:
            # Check the frozen binding against the tree before resolving the
            # request, so a deleted or replaced candidate is always reported as
            # an identity change rather than as whichever resolution error it
            # happened to cause.
            relative = expected_candidate.get("relativePath") \
                if isinstance(expected_candidate, dict) else None
            if not isinstance(relative, str):
                raise dynamic_target.TargetRequestError(
                    "CANDIDATE_IDENTITY_CHANGED: the candidate binding is malformed")
            try:
                current = dynamic_target.candidate_reference(source, relative)
            except (OSError, ValueError) as exc:
                raise dynamic_target.TargetRequestError(
                    "CANDIDATE_IDENTITY_CHANGED: the candidate bound at submission "
                    "is no longer a readable file under the source") from exc
            if current != expected_candidate:
                raise dynamic_target.TargetRequestError(
                    "CANDIDATE_IDENTITY_CHANGED: the candidate is not the one bound at submission")
        request = dynamic_target.resolve_request(
            manifest, order, source, target=target, candidate=candidate)
        if expected_candidate is not None and (
                request is None or request["candidate"] != expected_candidate):
            raise dynamic_target.TargetRequestError(
                "CANDIDATE_IDENTITY_CHANGED: the candidate is not the one bound at submission")
        if expected_source is not None and snapshot(source) != expected_source:
            raise dynamic_target.TargetRequestError(
                "SOURCE_IDENTITY_CHANGED: the source is not the one bound at submission")
        run_id, directory = self.store.create_run()
        receipt = {"schemaVersion": 1, "runId": run_id, "startedAt": now(), "finishedAt": None,
                   "adapter": {"id": manifest.id, "path": str(manifest.path), "sha256": manifest.manifest_hash,
                               "files": manifest.files}, "selectedGates": selected, "gates": [],
                   "source": snapshot(source), "configuration": {}, "status": "NOT_RUN",
                   "trust": "local-user-integrity", "historicalPassReuse": False}
        if request is not None:
            receipt["dynamicTarget"] = request
        try:
            receipt["configuration"] = configuration(manifest, env)
            resources = ["source:" + str(source)]
            # The lease names the directory the Adapter will actually write. Both
            # sides resolve it from the Adapter's machine file, so a lock and its
            # build cannot describe two different trees.
            build = build_directory(source, manifest, env)
            resources.append("build:" + str(build))
            variable, windows = device_lease_plan(manifest, order, source)
            device_resource = None
            device_root = None
            if variable is not None:
                device = env.get(variable, "")
                if not re.fullmatch(r"[0-9]+", device):
                    raise ValueError(f"explicit {variable} or --device-id authorization required")
                receipt["authorizedDeviceId"] = str(int(device))
                device_root = self._device_root()
                if not windows:
                    # Nothing declared a window, so the engine holds the card for
                    # the whole recipe exactly as it always has.
                    device_resource = "npu:" + str(int(device))
                else:
                    # The card is NOT leased here: each windowed recipe takes it
                    # itself, immediately around the work that touches it. The
                    # grant is recorded so a receipt states the same card the
                    # window would have to name.
                    receipt["deviceLease"] = {"mode": "windowed",
                                              "resource": "npu:" + str(int(device)),
                                              "gates": sorted(windows)}
            with ExitStack() as stack:
                lease_fds = stack.enter_context(acquire(resources, root=self.lock_root))
                if device_resource is not None:
                    lease_fds += stack.enter_context(
                        acquire([device_resource], root=device_root))
                # Capture after acquiring locks; another AgentGates run may just have finished.
                receipt["source"] = snapshot(source)
                if expected_source is not None and receipt["source"] != expected_source:
                    raise dynamic_target.TargetRequestError(
                        "SOURCE_IDENTITY_CHANGED: the source changed before the run acquired its locks")
                if request is not None and not candidate_matches(source, request):
                    raise dynamic_target.TargetRequestError(
                        "CANDIDATE_IDENTITY_CHANGED: the candidate changed before the run acquired its locks")
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
                    if is_quarantined_recipe(manifest.id, recipe.id, recipe.argv):
                        gate.update(
                            status="BLOCKED_ENV",
                            failures=[quarantine_reason(f"gate {gate_id!r} ({recipe.id})")],
                        )
                        continue
                    try:
                        self._gate(manifest, source, recipe, gate, receipt, env, directory, lease_fds,
                                   windows.get(gate_id), device_root)
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
                if request is not None and not candidate_matches(source, request):
                    for gate in receipt["gates"]:
                        if gate["status"] == "PASS":
                            gate.update(status="STALE", failures=["CANDIDATE_IDENTITY_CHANGED"])
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
        except dynamic_target.TargetRequestError:
            # A request that no longer describes the tree is a request refusal,
            # not a gate verdict: publishing BLOCKED_ENV would claim the machine
            # was at fault for a caller's changing candidate.
            raise
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

    def _gate(self, manifest, source, recipe, gate, receipt, env, directory, lease_fds,
              window_recipe=None, device_root=None):
        manifest.validate_files()
        request = receipt.get("dynamicTarget")
        bound = request if isinstance(request, dict) else None
        if recipe.dynamic_target:
            if bound is None:
                gate.update(status="INVALID_EVIDENCE", failures=["DYNAMIC_TARGET_MISSING"])
                return
        # A recipe that accepts the request consumes it when the run has one, so
        # its evidence must describe the same candidate bytes throughout.
        consumed = bound if dynamic_target.accepts_request(recipe) else None
        if consumed is not None and not candidate_matches(source, consumed):
            gate.update(status="STALE", failures=["CANDIDATE_IDENTITY_CHANGED"])
            return
        gate["sourceBefore"] = snapshot(source)
        if gate["sourceBefore"] != receipt["source"]:
            gate.update(status="STALE", failures=["SOURCE_IDENTITY_CHANGED"])
            return
        gate["executable"] = executable_identity(recipe.argv, recipe.cwd, env)
        if not recipe.config_check and manifest.id in ENV_KEYS:
            gate.update(status="BLOCKED_ENV", failures=["CONFIG_CHECK_MISSING"])
            return
        if recipe.config_check:
            check_env = dynamic_target.apply_environment(env, recipe, bound)
            check = execute(recipe.config_check, cwd=recipe.cwd, env=check_env,
                            timeout=config_check_timeout(recipe),
                            log_dir=directory / gate["gateId"] / "config", lease_fds=lease_fds)
            gate["configCheck"] = check
            if check["exitCode"] != 0 or check["timedOut"] or check["interrupted"]:
                # The Adapter's own refusal is the actionable reason (an
                # unconfigured machine, a card another job holds). Keep one
                # bounded line on the Gate so a reader does not have to open
                # the log to learn why the Gate never started. The failure
                # marker itself stays machine-readable and unparameterised.
                # A structured observation, when the Adapter publishes one,
                # adds the machine-readable category beside that human line.
                update = {"status": "BLOCKED_ENV", "failures": ["CONFIG_CHECK_FAILED"],
                          "configReason": config_reason(check)}
                observation = config_observation(check)
                if observation is not None:
                    update["configObservation"] = observation
                gate.update(**update)
                return
        manifest.validate_files()
        if consumed is not None and not candidate_matches(source, consumed):
            gate.update(status="STALE", failures=["CANDIDATE_IDENTITY_CHANGED"])
            return
        if snapshot(source) != gate["sourceBefore"]:
            gate.update(status="STALE", failures=["SOURCE_CHANGED_DURING_CONFIG_CHECK"])
            return
        if configuration(manifest, env) != receipt["configuration"]:
            gate.update(status="STALE", failures=["CONFIGURATION_IDENTITY_CHANGED"])
            return
        inputs = artifact_inputs(source, manifest, env)
        gate["artifactInputs"] = list(inputs.values())
        recipe_env = dynamic_target.apply_environment(env, recipe, bound)
        lease = None
        if window_recipe is not None:
            # The recipe opens the lease itself, so it is given the exact file the
            # engine would have locked. The whole-recipe lease is deliberately not
            # held: holding it here would deadlock the child that waits for it.
            root = device_root if device_root is not None else self._device_root()
            lease = {"resource": "npu:" + receipt["authorizedDeviceId"],
                     "path": str(lock_path("npu:" + receipt["authorizedDeviceId"], root))}
            recipe_env = dict(recipe_env)
            recipe_env.update({DEVICE_LEASE_PATH: lease["path"],
                               DEVICE_LEASE_RESOURCE: lease["resource"],
                               DEVICE_LEASE_WAIT: repr(float(window_recipe.device_window))})
        process = execute(recipe.argv, cwd=recipe.cwd, env=recipe_env, timeout=recipe.timeout,
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
            update = {"status": "BLOCKED_ENV", "failures": [f"BLOCKED_EXIT_CODE: {process['exitCode']}"],
                      "configReason": config_reason(process)}
            # A structured observation belongs to the configuration PREFLIGHT.
            # A recipe's own exit code describes execution, so it keeps only the
            # Adapter's human reason; letting the preflight category override an
            # execution-time classification would relabel a workload failure.
            gate.update(**update)
        if consumed is not None and not candidate_matches(source, consumed):
            # The bytes that ran are no longer the bytes the receipt binds, so
            # this Gate's evidence cannot describe one candidate.
            gate.update(status="STALE", failures=gate["failures"] + ["CANDIDATE_IDENTITY_CHANGED"])
        if lease is not None:
            # A windowed gate must SHOW its window. The engine cannot observe when
            # the child held the flock, so the child states it and this decides
            # whether the statement is admissible: the lease it names must be the
            # one it was given, both ends must exist in order, and occupancy must
            # be a defined value. Nothing published means no window was opened,
            # which is a failure - an unshown window is not a short one. Occupancy
            # itself is recorded and never gated: another tenant on the card is
            # normal, and must not become a failure.
            #
            # A gate already classified BLOCKED_ENV is the one exception, and it
            # is not an exemption from the rule: the Adapter reported that the
            # work could not start, so no window exists to be shown. Demanding one
            # would rewrite an honest resource refusal into INVALID_EVIDENCE, which
            # claims the evidence was bad rather than that nothing ran. The lease
            # is still accounted for: it must have been released, or a blocked run
            # would silently keep holding the card.
            if gate["status"] == "BLOCKED_ENV":
                gate["deviceWindow"] = {"resource": lease["resource"], "path": lease["path"],
                                        "startedAt": None, "releasedAt": None,
                                        "occupancy": "unknown", "waited": 0.0,
                                        "notOpened": True}
            else:
                window, failure = device_window(gate["observations"], lease)
                if failure is not None:
                    gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + [failure])
                else:
                    gate["deviceWindow"] = window
                    receipt.setdefault("deviceLeaseWindows", []).append({"gateId": gate["gateId"], **window})
        gate["sourceAfter"] = snapshot(source)
        if gate["sourceAfter"] != gate["sourceBefore"]:
            gate.update(status="STALE", failures=gate["failures"] + ["SOURCE_IDENTITY_CHANGED"])
        manifest.validate_files()
        try:
            gate["artifacts"] = collect_artifacts(gate["observations"], source)
            # Exit-code build gates still bind the actual products they created.
            if not gate["artifacts"]:
                gate["artifacts"] = list(artifact_inputs(source, manifest, env).values())
            if is_observation_only_recipe(manifest.id, recipe.id, gate["checks"]):
                for artifact in gate["artifacts"]:
                    if artifact["path"] in inputs and artifact != inputs[artifact["path"]]:
                        gate.update(status="STALE", failures=gate["failures"] + ["ARTIFACT_CHANGED_DURING_TEST"])
                    elif ".so" in Path(artifact["path"]).name and artifact["path"] not in inputs:
                        gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + ["ARTIFACT_PRETEST_IDENTITY_MISSING"])
            if any(key in gate["checks"] for key in ("loadedNativeSha256MatchesHost", "importedModuleSha256MatchesHost")) and not gate["artifacts"]:
                gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + ["ARTIFACT_IDENTITIES_MISSING"])
        except (OSError, ValueError) as exc:
            gate.update(status="INVALID_EVIDENCE", failures=gate["failures"] + [str(exc)])
