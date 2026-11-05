"""harness-run/1 device harness (H0-H2 adapter side).

Design: artifacts/run02/review/architect_h_device_harness_module_design_v1.md
(SHA 4f237b30..., frozen). This is a thin harness that consumes the canonical device
entry; it never opens another ENV path and never substitutes a generic add for an
unregistered workload.

Core invariants encoded here:
  * workload registry + SELECT_ONE / ALL_REQUESTED scope, reported separately;
  * a device run goes through the canonical recipe only (unknown workload ->
    UNSUPPORTED_WORKLOAD; a different/generic entry -> NON_CANONICAL_DEVICE_ENTRY);
  * only a pre-launch driver probe may fail over; once launch starts, any failure
    (timeout/error/numeric) stops this candidate and is judged in the original scope
    (LAUNCH_FAILED/SYNC_FAILED/NUMERIC_MISMATCH), never retried on another card;
  * phases and raw actual inputs+eps f32 bits+outputs are retained; four artifact
    identities (source/CCE/binary/load) are explicit (OBSERVED or UNOBSERVED);
  * FINISHED != PASS; local evidence is separate from any AgentGates service receipt;
  * CPU verify defaults to --skip-device; device needs allow_device=True.

The tests use fake drivers/owned temp dirs only; no card is touched.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

SCHEMA = "harness-run/1"
SELECT_ONE = "SELECT_ONE"
ALL_REQUESTED = "ALL_REQUESTED"
OBSERVED, UNOBSERVED = "OBSERVED", "UNOBSERVED"
ARTIFACT_KEYS = ("source", "cce", "binary", "load")


class HarnessError(RuntimeError):
    def __init__(self, code, message, source_id="", expected=None, actual=None, phase="harness"):
        self.code = code
        self.phase = phase
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s [%s] %s (source=%r expected=%r actual=%r)"
                         % (code, phase, message, source_id, expected, actual))

    def as_dict(self):
        return {"code": self.code, "phase": self.phase, "source_id": self.source_id,
                "expected": self.expected, "actual": self.actual}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    canonical_recipe: str
    x_mode: str
    output_spec: dict
    executor: str                  # named canonical entry, never "generic_add"
    allow_generic_fallback: bool = False


WORKLOADS: dict = {}


def register_workload(spec: WorkloadSpec) -> WorkloadSpec:
    if not spec.workload_id or not spec.canonical_recipe:
        raise HarnessError("UNSUPPORTED_WORKLOAD", "workload id/recipe missing",
                          "workload", "id+canonical recipe", spec)
    if spec.allow_generic_fallback or spec.executor in ("generic_add", "torch_add"):
        raise HarnessError("NON_CANONICAL_DEVICE_ENTRY", "generic add is not a workload entry",
                          spec.workload_id, "canonical executor", spec.executor)
    WORKLOADS[spec.workload_id] = spec
    return spec


def get_workload(workload_id: str) -> WorkloadSpec:
    spec = WORKLOADS.get(workload_id)
    if spec is None:
        raise HarnessError("UNSUPPORTED_WORKLOAD", "workload is not registered",
                          workload_id, sorted(WORKLOADS), workload_id)
    return spec


@dataclass(frozen=True)
class RunRequest:
    schema: str
    run_id: str
    parent_run: Optional[str]
    case_id: str
    workload_id: str
    adapter_commit: str
    manifest_hash: str
    recipe_id: str
    gate_id: str
    env_ref: dict
    profile_ref: dict
    source_ref: dict
    tir_ref: dict
    problem_ref: dict
    plan_ref: dict
    c_ref: dict
    x_mode: str
    x_artifact_refs: dict
    abi_ref: dict
    input_ref: dict
    protocol_ref: dict
    allow_device: bool = False


def make_run_request(case_spec, X_request, T_inputs, protocol_ref, env_ready_ref) -> RunRequest:
    """Build a request for the canonical device path (H0)."""
    case_spec = dict(case_spec or {})
    X_request = dict(X_request or {})
    T_inputs = dict(T_inputs or {})
    workload_id = str(X_request.get("workload_id") or case_spec.get("workload_id") or "")
    spec = get_workload(workload_id)
    for name, value in (("protocol_ref", protocol_ref), ("env_ready_ref", env_ready_ref)):
        if not value:
            raise HarnessError("EVIDENCE_MISSING", "%s is required" % name, name,
                              "non-empty reference", value)
    return RunRequest(
        schema=SCHEMA, run_id=str(case_spec.get("run_id") or "run-%d" % int(time.time() * 1e6)),
        parent_run=case_spec.get("parent_run"), case_id=str(case_spec.get("case_id", "case")),
        workload_id=workload_id, adapter_commit=str(env_ready_ref.get("adapter_commit", "")),
        manifest_hash=str(env_ready_ref.get("manifest_hash", "")),
        recipe_id=str(X_request.get("recipe_id", spec.canonical_recipe)),
        gate_id=str(X_request.get("gate_id", "")),
        env_ref=dict(env_ready_ref), profile_ref=dict(case_spec.get("profile_ref", {})),
        source_ref=dict(case_spec.get("source_ref", {})), tir_ref=dict(case_spec.get("tir_ref", {})),
        problem_ref=dict(case_spec.get("problem_ref", {})), plan_ref=dict(case_spec.get("plan_ref", {})),
        c_ref=dict(case_spec.get("c_ref", {})), x_mode=str(X_request.get("x_mode", spec.x_mode)),
        x_artifact_refs=dict(X_request.get("artifacts", {})),
        abi_ref=dict(X_request.get("abi", {})), input_ref=dict(T_inputs),
        protocol_ref=dict(protocol_ref), allow_device=bool(X_request.get("allow_device", False)))


@dataclass(frozen=True)
class AdapterRunHandle:
    recipe_id: str
    run_request: RunRequest


def dispatch_via_adapter(run_request: RunRequest, canonical_recipe: str,
                         adapter_call: Optional[Callable] = None) -> AdapterRunHandle:
    """The ONLY dispatch path: canonical recipe + registered workload."""
    spec = get_workload(run_request.workload_id)
    if canonical_recipe != spec.canonical_recipe:
        raise HarnessError("NON_CANONICAL_DEVICE_ENTRY",
                          "device run must use the canonical recipe",
                          run_request.workload_id, spec.canonical_recipe, canonical_recipe)
    if run_request.recipe_id != canonical_recipe:
        raise HarnessError("NON_CANONICAL_DEVICE_ENTRY", "request recipe differs",
                          run_request.run_id, canonical_recipe, run_request.recipe_id)
    if not run_request.allow_device:
        raise HarnessError("DEVICE_NOT_REQUESTED",
                          "CPU verify defaults to --skip-device; pass allow_device explicitly",
                          run_request.run_id, "allow_device=True", False)
    if adapter_call is not None:
        adapter_call(run_request)
    return AdapterRunHandle(canonical_recipe, run_request)


@dataclass(frozen=True)
class DeviceRequest:
    ordered_visible_physical_ids: tuple
    request_scope: str = SELECT_ONE
    start_logical_index: int = 0
    selection_policy: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.request_scope not in (SELECT_ONE, ALL_REQUESTED):
            raise HarnessError("DEVICE_REQUEST_INVALID", "unknown request_scope",
                              "scope", [SELECT_ONE, ALL_REQUESTED], self.request_scope)


@dataclass(frozen=True)
class SelectedDevice:
    original_visible: tuple
    logical_indices: tuple
    physical_ids: tuple
    scope: str
    selection_reason: str


@dataclass
class Attempt:
    attempt_id: int
    logical_index: int
    physical_id: int
    outcome: str
    timed_out: bool = False
    unreaped: bool = False
    wchan: Optional[str] = None
    rc: Optional[int] = None
    stderr_tail: str = ""
    backoff_seconds: Optional[float] = None


def select_device(device_request: DeviceRequest, verified_env, probe_policy) -> tuple:
    """Pre-launch only: shared ordered selection; only driver BUSY may fail over."""
    if not device_request.ordered_visible_physical_ids:
        raise HarnessError("DEVICE_REQUEST_INVALID", "visible list is empty", "visible", "nonempty", ())
    visible = list(device_request.ordered_visible_physical_ids)
    start = int(device_request.start_logical_index)
    if not 0 <= start < len(visible):
        raise HarnessError("DEVICE_REQUEST_INVALID", "start index outside visible list",
                          "start_logical_index", (0, len(visible) - 1), start)
    probe = probe_policy.get("probe")
    if probe is None:
        raise HarnessError("DEVICE_REQUEST_INVALID", "probe policy has no probe callable",
                          "selection_policy", "probe", None)
    attempts = []
    selected_ids = []
    if device_request.request_scope == SELECT_ONE:
        # Use the approved shared selection (same code path as ENV-05).
        try:
            from verify_environment import first_responsive_card  # noqa: PLC0415
            shared_attempts, selected, failure = first_responsive_card(
                list(range(start, len(visible))), visible, probe,
                float(device_request.selection_policy.get("backoff_seconds", 30.0)),
                sleeper=device_request.selection_policy.get("sleeper", time.sleep))
        except ImportError:  # pragma: no cover - only when scripts/ is not importable
            raise HarnessError("ENV_PROBE_UNAVAILABLE", "verify_environment not importable",
                              "probe", "canonical probe", None)
        attempts = shared_attempts
        if failure is not None:
            raise HarnessError(failure["code"], "pre-launch probe failed; no failover",
                              failure.get("physical_device_id"), "responsive", failure)
        if selected is None:
            return attempts, None
        selected_ids = [selected["physical_device_id"]]
        logical = [selected["logical_index"]]
    else:
        for logical in range(start, len(visible)):
            result = probe(visible[logical])
            classification = result.get("classification", "RESPONSIVE" if result.get("responsive") else "DEVICE_PROBE_FAILED")
            record = Attempt(len(attempts) + 1, logical, visible[logical], classification,
                             bool(result.get("timed_out")), bool(result.get("unreaped")),
                             result.get("wchan"), result.get("rc"), result.get("stderr_tail", ""))
            attempts.append(record)
            if classification == "RESPONSIVE":
                selected_ids.append(visible[logical])
        logical = [attempt.logical_index for attempt in attempts
                   if attempt.physical_id in selected_ids]
    if not selected_ids:
        return attempts, None
    selected = SelectedDevice(tuple(visible), tuple(logical), tuple(selected_ids),
                              device_request.request_scope,
                              "ordered visible list; selected %d" % len(selected_ids))
    return attempts, selected


@dataclass
class PhaseRecord:
    name: str
    status: str
    started: float
    ended: Optional[float] = None
    rc: Optional[int] = None
    error_code: Optional[str] = None
    detail: str = ""
    raw_stdout_path: Optional[str] = None
    raw_stderr_path: Optional[str] = None

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    phases: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    _counter: int = 0

    def record(self, name, status, *, rc=None, error_code=None, detail="", start=None):
        self._counter += 1
        now = time.time()
        record = PhaseRecord(name, status, start if start is not None else now,
                             now if status != "START" else None, rc, error_code, detail)
        self.phases.append(record)
        return record

    def write_raw(self, name: str, payload: bytes) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path


def execute_attempt(selected: SelectedDevice, x_artifacts, abi, actual_inputs,
                    run_context: RunContext, launch_fn=None) -> list:
    """Execute one already-selected candidate set. Never selects another card.

    Once "launch" is recorded as START, ANY exception is LAUNCH_FAILED and no
    failover is attempted; a missing output is READBACK_MISSING/OUTPUT_MISSING.
    """
    if launch_fn is None:
        raise HarnessError("LAUNCH_FAILED", "no launch function", "execute", "callable", None)
    if not selected or not selected.physical_ids:
        raise HarnessError("DEVICE_NOT_SELECTED", "no responsive device", "execute",
                          "selected device", None)
    # raw actual inputs + eps f32 bits are recorded BEFORE launch
    for name, item in sorted((actual_inputs or {}).items()):
        payload = item["bytes"] if isinstance(item, dict) else bytes(item)
        path = run_context.write_raw("inputs/%s.bin" % name, payload)
        run_context.inputs.append({"name": name, "path": str(path),
                                   "nbytes": len(payload), "sha256": sha256_bytes(payload),
                                   "dtype": item.get("dtype") if isinstance(item, dict) else None,
                                   "shape": item.get("shape") if isinstance(item, dict) else None})
    run_context.inputs.append({"name": "eps", "f32_bits": int(actual_inputs.get("eps_f32_bits"))
                               if isinstance(actual_inputs.get("eps_f32_bits"), int) else None,
                               "dtype": "f32"})
    produced = []
    for physical_id in selected.physical_ids:
        run_context.record("launch", "START")
        try:
            result = launch_fn(physical_id, x_artifacts, abi)
        except Exception as exc:  # noqa: BLE001
            run_context.record("launch", "FAILED", error_code="LAUNCH_FAILED",
                               detail=repr(exc), start=time.time())
            raise HarnessError("LAUNCH_FAILED", "kernel launch failed; no failover after launch",
                              str(physical_id), "successful launch", repr(exc), phase="launch")
        run_context.record("launch", "PASS", start=time.time())
        run_context.record("sync", "START")
        if not result.get("synced", True):
            run_context.record("sync", "FAILED", error_code="SYNC_FAILED",
                               detail="synchronize did not complete")
            raise HarnessError("SYNC_FAILED", "synchronize failed", str(physical_id),
                              True, False, phase="sync")
        run_context.record("sync", "PASS")
        run_context.record("readback", "START")
        payload = result.get("output")
        if payload is None or not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
            run_context.record("readback", "FAILED", error_code="READBACK_MISSING",
                               detail="no raw output bytes")
            raise HarnessError("OUTPUT_MISSING", "readback produced no output", str(physical_id),
                              "nonempty bytes", payload, phase="readback")
        out_path = run_context.write_raw("outputs/dev%s.bin" % physical_id, bytes(payload))
        output = {"physical_id": physical_id, "path": str(out_path),
                  "nbytes": len(payload), "sha256": sha256_bytes(bytes(payload)),
                  "dtype": result.get("dtype"), "shape": result.get("shape")}
        run_context.outputs.append(output)
        run_context.record("readback", "PASS")
        produced.append(output)
    return produced


def collect_evidence(run_context: RunContext, phase_events, outputs,
                     artifact_identity=None) -> dict:
    """Persist raw phase/input/output refs and the four artifact identities."""
    artifact_identity = dict(artifact_identity or {})
    artifacts = {}
    for key in ARTIFACT_KEYS:
        value = artifact_identity.get(key)
        if isinstance(value, dict):
            artifacts[key] = {"status": value.get("status", OBSERVED),
                              "path": value.get("path"), "sha256": value.get("sha256")}
        elif value:
            artifacts[key] = {"status": OBSERVED, "path": str(value),
                              "sha256": sha256_file(Path(value)) if Path(value).is_file() else None}
        else:
            artifacts[key] = {"status": UNOBSERVED, "path": None, "sha256": None}
    evidence = {
        "schema": SCHEMA, "run_id": run_context.run_id, "run_dir": str(run_context.run_dir),
        "phases": [phase.as_dict() if hasattr(phase, "as_dict") else dict(phase)
                   for phase in (phase_events or run_context.phases)],
        "attempts": [attempt.__dict__ if hasattr(attempt, "__dict__") else dict(attempt)
                     for attempt in run_context.attempts],
        "inputs": run_context.inputs, "outputs": run_context.outputs,
        "artifacts": artifacts,
    }
    evidence["phase_digest"] = sha256_bytes(
        json.dumps(evidence["phases"], sort_keys=True, default=str).encode())
    evidence["content_digest"] = sha256_bytes(
        json.dumps({key: evidence[key] for key in ("inputs", "outputs", "artifacts")},
                   sort_keys=True, default=str).encode())
    manifest_path = run_context.run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(evidence, indent=1, sort_keys=True, default=str) + "\n")
    evidence["manifest_path"] = str(manifest_path)
    return evidence


def evaluate_run(request: RunRequest, evidence: dict, t_report=None) -> dict:
    """FINISHED != PASS. A finished run can fail every required condition below."""
    phases = list(evidence.get("phases", []) or [])
    by_name = {}
    for phase in phases:
        by_name[phase.get("name")] = phase
    # finished means the run is over (raw evidence was written), NOT that it passed.
    finished = bool(evidence.get("manifest_path"))
    reason = None
    passed = False
    failed_phase = next((phase for phase in phases if phase.get("status") == "FAILED"), None)
    if failed_phase is not None:
        reason = failed_phase.get("error_code") or "PHASE_FAILED"
    elif not phases or any(name not in by_name for name in ("launch", "sync", "readback")):
        reason = "COMPLETION_MISSING"
    elif len(evidence.get("outputs", []) or []) == 0:
        reason = "OUTPUT_MISSING"
    elif request.input_ref and not evidence.get("inputs"):
        reason = "INPUT_IDENTITY_MISMATCH"
    elif any(evidence.get("artifacts", {}).get(key, {}).get("status") != OBSERVED
             for key in ARTIFACT_KEYS):
        reason = "ARTIFACT_IDENTITY_UNOBSERVED"
    elif request.recipe_id == "":
        reason = "NON_CANONICAL_DEVICE_ENTRY"
    elif t_report is not None and t_report.get("status") != "PASS":
        reason = "NUMERIC_MISMATCH"
    else:
        # scope/coverage is separate from the failed-phase check
        expected = request.x_artifact_refs.get("coverage", {})
        if expected.get("scope") == ALL_REQUESTED:
            expected_ids = tuple(expected.get("physical_ids", ()))
            got_ids = tuple(output.get("physical_id") for output in evidence.get("outputs", []))
            if set(expected_ids) != set(got_ids) or len(got_ids) != len(expected_ids):
                reason = "DEVICE_COVERAGE_MISMATCH"
        if reason is None:
            passed = True
    return {
        "schema": SCHEMA, "run_id": request.run_id, "scope": request.x_artifact_refs.get(
            "coverage", {}).get("scope", SELECT_ONE),
        "finished": bool(finished), "passed": bool(passed),
        "verdict": "PASS" if passed else ("NOT_RUN" if reason == "DEVICE_NOT_SELECTED" else "FAIL"),
        "reason_code": reason, "required_phases": ["launch", "sync", "readback"],
        "manifest_path": evidence.get("manifest_path"), "local_only": True,
        "service_receipt": None,
    }


def replay_run_result(request: RunRequest, manifest: dict) -> dict:
    """Re-evaluate a saved manifest; hashes are checked from the raw files, not trusted."""
    for item in list(manifest.get("inputs", [])) + list(manifest.get("outputs", [])):
        path = item.get("path")
        if path and Path(path).is_file():
            actual = sha256_file(Path(path))
            if item.get("sha256") and actual != item["sha256"]:
                raise HarnessError("EVIDENCE_MISSING", "raw file changed since manifest",
                                  str(path), item["sha256"], actual)
    return evaluate_run(request, manifest)


def default_cpu_verify_args() -> list:
    """CPU verify never touches the card unless a device recipe is explicitly asked."""
    return ["--side", "recipe", "--skip-device"]


__all__ = ["SCHEMA", "SELECT_ONE", "ALL_REQUESTED", "WorkloadSpec", "register_workload",
           "get_workload", "RunRequest", "make_run_request", "dispatch_via_adapter",
           "DeviceRequest", "select_device", "execute_attempt", "RunContext",
           "collect_evidence", "evaluate_run", "replay_run_result",
           "default_cpu_verify_args", "HarnessError", "ARTIFACT_KEYS"]
