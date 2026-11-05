#!/usr/bin/env python3
"""One entry point that verifies THE deployed environment, end to end.

Why one entry point: the pieces (CANN, the wheel, configure, a compile, a card) were
verified by separate scripts. Separate scripts let a run report five greens that were
measured against five different environments. This program measures ONE environment,
in ONE run directory, and writes ONE JSON.

Phase DAG (each phase names what it depends on, and is NOT RUN if that dependency
failed):

    cann -> layers -> configure_idempotent -> tf_cpu_tools -> compile_smoke -> device_smoke

tf_cpu_tools is ENV-04's "CPU tools really run" phase: a tiny CHECKED TileFoundry
program goes checked -> analyze -> finalize, and the TYPED artefact plus the
structured reports decide the verdict. /bin/true -- or any step that exits 0 and
produces nothing -- FAILS.

Hard rules encoded here:
  * a phase that FINISHED is not a phase that PASSED: every phase must produce the
    evidence it claims (a compile that emits no source FAILS, an exit 0 with no
    report FAILS);
  * a busy card is RECORDED, never waited for and never a refusal;
  * the npu-smi "after" snapshot happens exactly once, in a finally block;
  * the after snapshot is taken even when the device phase never ran, so an empty
    "after" file can never be mistaken for a clean run;
  * the canonical team lock is taken non-blocking; if another job holds it the run is
    BLOCKED, not silently re-entered.

Exit codes: 0 PASS | 1 FAIL | 2 CONFIG_ERROR | 75 BLOCKED_ENV | 97 NOT_RUN.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
sys.path.insert(0, str(HERE))
import env_context as E  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_BLOCKED = 75
EXIT_NOT_RUN = 97

#: The canonical TEAM lock. /home is where every team job (and the container's root
#: side) can write, so both halves agree on one path. A per-user path would silently
#: not serialise two team members with different accounts.
TEAM_LOCK_PATH = Path("/home/.tf_device0.lock")  # no-machine-path: allow
#: TF_DEVICE_LOCK_PATH is an EXPLICIT override for TESTS and isolation only. A run
#: that uses it is not using the team lock, and the evidence says so: "the lease
#: mechanism works" is a different claim from "the team was serialised".
TEAM_LOCK = Path(os.environ.get("TF_DEVICE_LOCK_PATH") or TEAM_LOCK_PATH)

# Imported at MODULE scope on purpose. The eager builder resolves `T` from the
# defining scope; a function-local `import tilelang.ascend.language as T` inside the
# kernel factory made the rewrite fail with "Attribute assignment not supported yet"
# (measured), which reads like a language limitation and is really a scoping mismatch.
try:
    import tilelang.ascend.language as T  # noqa: N816
except Exception:  # noqa: BLE001  -- reported by the layers phase, not here
    T = None
CANONICAL_SET_ENV = "/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh"
N_LANES = 64

#: The smallest program that still exercises the whole checked -> analyze ->
#: finalize chain: one 64-lane AIV mesh with two T.copy calls. It is deliberately
#: adapter-owned rather than borrowed from the TileFoundry examples: ENV-04 must
#: prove THIS adapter's deployment can run a checked program, not that one
#: example file happens to be present.
TINY_CHECKED_PROGRAM = '''"""The adapter's smallest checked TileFoundry program (ENV-04)."""

from tilefoundry.dsl import *
from tilefoundry_ascend.tf_target import Ascend950PRTarget

N = 64


@module(entry="copy1", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class CopySlice:
    @func
    def copy1(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(N,), "f32", (N,), "smem"])
            staged = tf.schedule((x, ub), op=T.copy())
            out = tf.empty(Tensor[(N,), "f32", (N,), "gmem"])
            return tf.schedule((staged, out), op=T.copy())
'''

#: Runs INSIDE the deployment interpreter. checked_program is the frontend's own
#: checked entry point: it loads the module, binds dims and runs the semantic
#: checks, so a structured result here means the program was accepted, not merely
#: imported.
CHECKED_HELPER = '''"""Load one program through the checked frontend and print what it accepted."""
import json
import sys

from tilefoundry.cli.source import checked_program

program = checked_program(sys.argv[1], None, command="verify_environment")
module = program.module
print(json.dumps({
    "module": getattr(module, "name", None),
    "entry": getattr(program.entry, "name", None),
    "functions": len(getattr(module, "functions", ()) or ()),
}))
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_kernel_artifacts(kernel, out_dir, facts: dict) -> None:
    """Export the compiled kernel and hash the REAL bytes on disk.

    "The compile exited 0" is not an artefact. This writes the Ascend C source and
    the linked library and records their sha256, along with the device/host sources
    the adapter produced. ENV-05 asks for the produced artifact hash; that has to be
    a hash of a file, not a flag.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for label, filename, kind in (
            ("ascend_c_source", "kernel_source.c", "sources"),
            ("linked_library", "kernel.lib", "library")):
        path = out_dir / filename
        try:
            if kind == "sources":
                kernel.export_sources(str(path))
            else:
                kernel.export_library(str(path))
        except Exception as exc:  # noqa: BLE001
            artifacts[label] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        if path.is_file():
            data = path.read_bytes()
            artifacts[label] = {"path": str(path), "bytes": len(data),
                                "sha256": sha256_bytes(data)}
        else:
            artifacts[label] = {"error": "export produced no file at %s" % path}
    facts["compiled_artifacts"] = artifacts

    def _source(label, getter):
        try:
            text = getter() or ""
        except Exception as exc:  # noqa: BLE001
            facts["%s_error" % label] = "%s: %s" % (type(exc).__name__, exc)
            return
        facts["%s_chars" % label] = len(text)
        facts["%s_sha256" % label] = sha256_bytes(text.encode())

    _source("devicesource", lambda: kernel.adapter.get_device_source())
    _source("hostsource", lambda: kernel.adapter.get_host_source())
    _source("kernelsource", lambda: getattr(kernel, "kernel_source", "") or "")


#: Measured on THIS machine (all numbers are observations, not guesses):
#:   healthy npu-smi: host 0.77/0.78 s, container 1.02/1.00 s; a full device
#:   phase saw the driver answer in 1.4 s; the non-blocking health probe's worst
#:   HEALTHY observation was 6 s. A D-state probe (wchan devdrv_sync_non_tran)
#:   took >20 s and could not be reclaimed even after SIGKILL -- it is unbounded,
#:   so waiting longer tells us nothing.
#: Therefore ONE card attempt is bounded at 20 s (separates <=6 s healthy from
#: >20 s wedged), and a failed attempt waits 30 s before the NEXT card -- one
#: natural window for transient congestion to clear without burning the run.
#: The attempt count is bounded by the visible list (2 cards today), so the worst
#: case is 2 x (20 + 30) ~= 100 s. Both numbers are env-overridable, in seconds.
DEVICE_ATTEMPT_TIMEOUT_DEFAULT = 20.0
DEVICE_BACKOFF_DEFAULT = 30.0


def bounded_run(argv, timeout, kill_grace=5, env=None):
    """Run *argv* with a bound that survives an UNKILLABLE child.

    Measured hang this exists for: npu-smi blocked in uninterruptible sleep (D) behind
    a neighbouring tenant's device job. subprocess.run(timeout=...) kills the child and
    then WAITS for it, so a D-state child makes the "timeout" unbounded. Here the child
    gets its own process group, the group is killed, and if it still has not exited the
    pipes are closed and the child is abandoned -- the caller records a busy snapshot
    and moves on. An unreaped child is reported, never hidden.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True, env=env)
    child_pid = proc.pid
    try:
        out, err = proc.communicate(timeout=timeout)
        return {"rc": proc.returncode, "out": out, "err": err, "pid": child_pid,
                "timed_out": False, "unreaped": False, "wchan": None}
    except subprocess.TimeoutExpired:
        # Read the kernel wait channel BEFORE killing: it is the difference between
        # "the driver queue is wedged" (D state) and "the command is merely slow".
        wchan = None
        try:
            wchan = Path("/proc/%d/wchan" % proc.pid).read_text().strip() or None
        except OSError:
            wchan = None
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            out, err = proc.communicate(timeout=kill_grace)
            return {"rc": proc.returncode, "out": out, "err": err, "pid": child_pid,
                    "timed_out": True, "unreaped": False, "wchan": wchan}
        except subprocess.TimeoutExpired:
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            UNREAPED_CHILDREN.append({"pid": child_pid, "argv": list(argv),
                                     "wchan": wchan,
                                     "recorded_at": time.strftime("%Y%m%dT%H%M%S")})
            return {"rc": None, "out": "", "err": "", "pid": child_pid,
                    "timed_out": True, "unreaped": True, "wchan": wchan}


def _json_document(text):
    """The JSON object a step printed, or None. Prose is not a structured result."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return value
    # Some tools print a banner first; accept the last complete line that parses.
    for line in reversed(text.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def run_tf_cpu_chain(tf_cmd, program, workdir, env, timeout: int = 180) -> dict:
    """checked -> analyze -> finalize, with the PRODUCT deciding the verdict.

    /bin/true exits 0 and produces nothing, so an exit code is never enough at any
    step:
      * checked must return a structured object with entrance and function count;
      * analyze must return a JSON report naming what it executed;
      * finalize must WRITE typed TIR (bytes, sha256, @prim_func, tensor_view).
    Every step is bounded and reported; no step is retried.
    """
    workdir = Path(workdir)
    helper = workdir / "checked_program_helper.py"
    helper.write_text(CHECKED_HELPER)
    selection = "%s:CopySlice" % program
    facts = {"command": list(tf_cmd), "program": str(program),
             "interpreter": str(program), "steps": [],
             "pythonpath": (env or {}).get("PYTHONPATH")}

    def step(name, argv):
        started = time.time()
        result = bounded_run(argv, timeout, env=env)
        record = {"step": name, "argv": argv, "rc": result["rc"],
                  "timed_out": result["timed_out"],
                  "unreaped": result["unreaped"],
                  "seconds": round(time.time() - started, 2)}
        facts["steps"].append(record)
        return result, record

    # ---- 1. checked
    result, record = step("checked", list(tf_cmd) + ["-B", str(helper), selection])
    checked = _json_document(result.get("out"))
    record["structured"] = checked is not None
    if result["timed_out"]:
        return {"ok": False, "code": "TF_CHECKED_TIMEOUT",
                "detail": "checked_program exceeded %ss" % timeout, "facts": facts}
    if result["rc"] != 0:
        return {"ok": False, "code": "TF_CHECKED_FAILED",
                "detail": (result.get("err") or result.get("out") or "").strip()[-400:],
                "facts": facts}
    if (checked is None
            or not isinstance(checked.get("entry"), str) or not checked["entry"]
            or type(checked.get("functions")) is not int or checked["functions"] < 1):
        # exit 0 with no accepted program is the /bin/true case, and it is a FAIL.
        return {"ok": False, "code": "TF_CHECKED_NO_STRUCTURED_RESULT",
                "detail": "checked_program exited 0 but produced no structured "
                          "result: %r" % ((result.get("out") or "").strip()[-300:],),
                "facts": facts}
    record["module"] = checked.get("module")

    # ---- 2. analyze
    result, record = step("analyze", list(tf_cmd) + [
        "-B", "-m", "tilefoundry.cli", "analyze", selection,
        "--compute-cost", "--json"])
    report = _json_document(result.get("out"))
    record["structured"] = report is not None
    if result["timed_out"]:
        return {"ok": False, "code": "TF_ANALYZE_TIMEOUT",
                "detail": "analyze exceeded %ss" % timeout, "facts": facts}
    if result["rc"] != 0:
        return {"ok": False, "code": "TF_ANALYZE_FAILED",
                "detail": (result.get("err") or result.get("out") or "").strip()[-400:],
                "facts": facts}
    if (report is None
            or not isinstance(report.get("executed"), list) or not report["executed"]
            or not isinstance(report.get("functions"), list) or not report["functions"]):
        return {"ok": False, "code": "TF_ANALYZE_NO_STRUCTURED_RESULT",
                "detail": "analyze exited 0 but produced no useful JSON report: %r"
                          % ((result.get("out") or "").strip()[-300:],),
                "facts": facts}
    analyze_path = workdir / "analyze.json"
    analyze_path.write_text(result.get("out") or "")
    facts["analyze"] = {"file": str(analyze_path),
                        "executed": report["executed"],
                        "functions": len(report["functions"]),
                        "sha256": sha256_bytes((result.get("out") or "").encode())}

    # ---- 3. finalize: the TYPED ARTEFACT is the deliverable
    tir = workdir / "tir.py"
    if tir.exists():
        tir.unlink()  # a stale artefact from a previous attempt must not be accepted
    result, record = step("finalize", list(tf_cmd) + [
        "-B", "-m", "tilefoundry.cli", "schedule", "finalize", selection, str(tir)])
    text = tir.read_text() if tir.is_file() else ""
    typed = (len(text) >= 200 and "@prim_func" in text and "tensor_view" in text)
    record.update({"artifact": str(tir), "bytes": len(text), "typed": typed})
    facts["finalized_artifact"] = {
        "path": str(tir), "bytes": len(text), "typed": typed,
        "sha256": sha256_bytes(text.encode()) if text else None,
        "contains_prim_func": "@prim_func" in text,
        "contains_tensor_view": "tensor_view" in text,
    }
    if result["timed_out"]:
        return {"ok": False, "code": "TF_FINALIZE_TIMEOUT",
                "detail": "finalize exceeded %ss" % timeout, "facts": facts}
    if result["rc"] != 0:
        return {"ok": False, "code": "TF_FINALIZE_FAILED",
                "detail": (result.get("err") or result.get("out") or "").strip()[-400:],
                "facts": facts}
    if not typed:
        # exit 0 and nothing written is exactly the false-positive this refuses.
        return {"ok": False, "code": "TF_FINALIZE_NO_TYPED_ARTIFACT",
                "detail": "finalize exited 0 but wrote no typed TIR (%d bytes, "
                          "@prim_func=%s, tensor_view=%s)"
                          % (len(text), "@prim_func" in text, "tensor_view" in text),
                "facts": facts}
    return {"ok": True, "code": None, "detail": None, "facts": facts}


BUSY_CODE = "BUSY_DEVICE_NOT_RESPONDING"

#: Children that had to be ABANDONED (still unreaped, possibly D-state). Recorded in
#: the evidence so a reader knows a process of ours may still be alive after exit.
UNREAPED_CHILDREN = []


def device_responsiveness(timeout: int = 20) -> dict:
    """A GLOBAL driver pulse (not per-card). Used only as a diagnostic record.

    The device phase itself probes each visible card individually, because a D-state
    wedged queue can be attached to ONE card while another still answers (user
    policy). This helper is kept for the bounded global observation; it never
    decides a verdict by itself.
    """
    started = time.time()
    result = bounded_run(["npu-smi", "info"], timeout=timeout)
    facts = {"probe": "npu-smi info", "timeout_s": timeout,
             "timed_out": result["timed_out"], "unreaped": result["unreaped"],
             "rc": result["rc"], "seconds": round(time.time() - started, 2),
             "child_pid": result.get("pid"), "wchan": result.get("wchan")}
    responsive = (not result["timed_out"]) and result["rc"] == 0 and bool(result["out"].strip())
    facts["responsive"] = responsive
    return facts


def acquire_team_lock(lock_path):
    """Take the team lease non-blocking and return the open handle.

    An open handle is the whole lock: the kernel releases the flock when the fd
    closes, including on os._exit. tests/test_verify_evidence_integrity.py proves
    that a completed verify process leaves the lock re-acquirable.
    """
    import fcntl

    handle = open(lock_path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


#: A wchan proves the child is parked in a DRIVER wait only when it NAMES the
#: driver. A missing or unrelated wchan means the timeout cannot be attributed to
#: the driver, so it must NOT become BUSY_DEVICE_NOT_RESPONDING (and must not switch).
_WCHAN_MARKERS = ("devdrv", "ascend", "npu", "hisi", "drv")


def wchan_supports_driver_wait(wchan) -> bool:
    text = str(wchan or "").strip().lower()
    return bool(text) and any(marker in text for marker in _WCHAN_MARKERS)


def probe_card(physical_id: int, timeout: float) -> dict:
    """One BOUNDED call that asks whether THIS card's driver queue answers.

    Classification is strict (architect H addendum):
      * timed_out + unreaped + a DRIVER wchan  -> BUSY_DEVICE_NOT_RESPONDING
        (the only case eligible for backoff/failover);
      * timed_out + unreaped, no usable wchan  -> DEVICE_PROBE_UNATTRIBUTED (no switch);
      * timed_out but the child died           -> DEVICE_PROBE_TIMEOUT (no switch);
      * fast rc != 0 (missing library/permission/command) -> DEVICE_PROBE_FAILED
        (CONFIG/BLOCKED, with rc and stderr; NO backoff, NO next card);
      * rc == 0 but empty output               -> DEVICE_PROBE_EMPTY_OUTPUT (no switch);
      * otherwise                              -> RESPONSIVE (use immediately).
    "Busy with someone else's job" is normal shared use and is not any of these.
    """
    started = time.time()
    forced = None
    spec = ""
    if (os.environ.get("TFNU_ALLOW_TEST_SEAMS") or "").strip() == "1":
        spec = (os.environ.get("TFNU_PROBE_OVERRIDE") or "").strip()
    if spec:
        for part in spec.split(","):
            key, _, value = part.partition("=")
            if key.strip().isdigit() and int(key.strip()) == int(physical_id):
                forced = value.strip()
    if forced == "unresponsive":
        result = {"rc": None, "out": "", "err": "", "pid": -1,
                  "timed_out": True, "unreaped": True,
                  "wchan": "devdrv_sync_non_tran"}
        simulated = True
    elif forced == "unattributed":
        result = {"rc": None, "out": "", "err": "", "pid": -1,
                  "timed_out": True, "unreaped": True, "wchan": None}
        simulated = True
    elif forced == "fail":
        result = {"rc": 2, "out": "", "err": "libc_sec.so: cannot open shared object file",
                  "pid": -1, "timed_out": False, "unreaped": False, "wchan": None}
        simulated = True
    elif forced == "empty":
        result = {"rc": 0, "out": "   \n", "err": "", "pid": -1,
                  "timed_out": False, "unreaped": False, "wchan": None}
        simulated = True
    elif forced == "responsive":
        result = {"rc": 0, "out": "forced-responsive", "err": "", "pid": -1,
                  "timed_out": False, "unreaped": False, "wchan": None}
        simulated = True
    else:
        result = bounded_run(
            ["npu-smi", "info", "-t", "board", "-i", str(physical_id)], timeout=timeout)
        simulated = False
    output = result["out"] or ""
    stderr = result["err"] or ""
    wchan = result.get("wchan")
    if result["timed_out"]:
        if result["unreaped"] and wchan_supports_driver_wait(wchan):
            classification = BUSY_CODE
        elif result["unreaped"]:
            classification = "DEVICE_PROBE_UNATTRIBUTED"
        else:
            classification = "DEVICE_PROBE_TIMEOUT"
    elif result["rc"] != 0:
        classification = "DEVICE_PROBE_FAILED"
    elif not output.strip():
        classification = "DEVICE_PROBE_EMPTY_OUTPUT"
    else:
        classification = "RESPONSIVE"
    return {"physical_id": int(physical_id), "simulated": simulated,
            "probe": "npu-smi info -t board -i %s" % physical_id,
            "responsive": classification == "RESPONSIVE",
            "classification": classification,
            "timed_out": result["timed_out"], "unreaped": result["unreaped"],
            "rc": result["rc"], "pid": result.get("pid"),
            "wchan": wchan, "stderr_tail": stderr.strip()[-400:],
            "seconds": round(time.time() - started, 2)}


def first_responsive_card(candidates, visible, probe, backoff_seconds,
                          sleeper=time.sleep) -> tuple:
    """Walk the VISIBLE list in order, one bounded probe per card.

    Only BUSY_DEVICE_NOT_RESPONDING (driver timeout + unreaped + driver wchan) may
    back off and move to the next card. Any other non-responsive classification is a
    FAILURE for this run: it stops immediately (no backoff, no next card) and is
    returned as the third element. A responsive card is used immediately, even when
    another tenant is using it (normal shared use is not a refusal).

    Returns (attempts, selected, failure); selected is None and failure is None iff
    every card was BUSY (all unresponsive).
    """
    attempts = []
    selected = None
    failure = None
    for position, logical_index in enumerate(candidates):
        physical_id = visible[logical_index]
        result = probe(physical_id)
        attempt = {"attempt": position + 1, "logical_index": logical_index, **result}
        classification = result.get("classification", BUSY_CODE if not result.get("responsive") else "RESPONSIVE")
        attempt["outcome"] = classification
        attempts.append(attempt)
        if classification == "RESPONSIVE":
            selected = {
                "logical_index": logical_index,
                "physical_device_id": physical_id,
                "selection_reason": (
                    "explicit --device-index (and later cards, if it is wedged)"
                    if candidates != list(range(len(visible))) else
                    "first RESPONSIVE entry of the ordered visible list"),
            }
            break
        if classification != BUSY_CODE:
            failure = {
                "code": classification,
                "physical_device_id": physical_id,
                "rc": result.get("rc"),
                "timed_out": result.get("timed_out"),
                "unreaped": result.get("unreaped"),
                "wchan": result.get("wchan"),
                "stderr_tail": result.get("stderr_tail", ""),
            }
            break
        if position < len(candidates) - 1:
            attempt["backoff_seconds"] = float(backoff_seconds)
            sleeper(float(backoff_seconds))
    return attempts, selected, failure


def smi(tag: str, out_dir: Path) -> dict:
    """Bounded npu-smi. A busy card makes npu-smi HANG; that is recorded, not waited on."""
    started = time.time()
    result = bounded_run(["npu-smi", "info"], timeout=25)
    elapsed = round(time.time() - started, 2)
    (out_dir / ("smi_%s.txt" % tag)).write_text(
        result["out"] or ("TIMED OUT after 25s (device busy); unreaped=%s\n"
                          % result["unreaped"]))
    return {"tag": tag, "rc": result["rc"], "bytes": len(result["out"]),
            "timed_out": result["timed_out"], "unreaped": result["unreaped"],
            "seconds": elapsed, "file": "smi_%s.txt" % tag,
            "code": "SMI_TIMEOUT" if result["timed_out"] else None,
            "note": "busy is recorded, never waited on, never a refusal"}


class PhaseTimeout(Exception):
    """The phase exceeded its hard bound. A named code, not an outer kill."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise PhaseTimeout("phase exceeded --phase-timeout")


class Phase:
    def __init__(self, pid, depends_on, fn):
        self.id = pid
        self.depends_on = depends_on
        self.fn = fn
        self.status = None
        self.code = None
        self.detail = None
        self.facts = {}
        self.seconds = None
        self.exit_hint = EXIT_FAIL

    def as_dict(self):
        return {"id": self.id, "depends_on": self.depends_on, "status": self.status,
                "code": self.code, "detail": self.detail, "facts": self.facts,
                "seconds": self.seconds}


def distribution_versions() -> dict:
    import importlib.metadata as md

    return {dist.metadata["Name"]: dist.version for dist in md.distributions()}


def device_selection(requested, visible_env):
    """Resolve a logical index inside the ordered visible list. No card is assumed."""
    visible = [int(x) for x in str(visible_env).split(",") if x.strip() != ""]
    if not visible:
        return None, "DEVICE_ID_MISSING"
    if requested is None:
        return {"visible_list": visible, "logical_index": 0,
                "physical_device_id": visible[0],
                "selection_reason": "first entry of the ordered visible list"}, None
    if not 0 <= requested < len(visible):
        return None, "DEVICE_INDEX_NOT_VISIBLE"
    return {"visible_list": visible, "logical_index": requested,
            "physical_device_id": visible[requested],
            "selection_reason": "explicit --device-index"}, None


def vadd_one(n: int):
    if T is None:
        raise RuntimeError("TILELANG_LANGUAGE_UNAVAILABLE: tilelang.ascend.language did not import")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1) as bx:
            a = T.alloc_shared((n,), "float32")
            T.copy(A[0:n], a)
            with T.SimtVF(threads=n):
                for i in T.Parallel(n):
                    a[i] = a[i] + 1.0
            T.copy(a, C[0:n])

    return main


#: Set on the re-exec so the deployment interpreter does not re-exec again.
REEXEC_ENV = "TFNU_VERIFY_REEXEC"
LAUNCHED_BY_ENV = "TFNU_VERIFY_LAUNCHED_BY"


def ensure_deployment_interpreter() -> None:
    """Continue under the DEPLOYMENT interpreter, whoever launched this file.

    The canonical recipe runs `python3 -B .../verify_environment.py`, and on this
    host `python3` is the system interpreter, which has no deployed tilelang. The
    environment contract is about the DEPLOYMENT interpreter, so the script picks
    it from env_context (never from a coincidental PATH entry) and re-execs itself
    into it. The launch interpreter is recorded, so "which python ran this" is a
    fact rather than an assumption.

    A missing deployment interpreter is NOT repaired here: the caller is left with
    the old interpreter, and the phases fail by name (TILELANG_IMPORT_FAILED etc.)
    rather than this check pretending an install happened.
    """
    if os.environ.get(REEXEC_ENV) == "1":
        return
    target = E.interpreter()
    try:
        same = target.exists() and Path(sys.executable).resolve() == target.resolve()
    except OSError:
        same = False
    if same or not target.exists():
        return
    env = dict(os.environ)
    env[REEXEC_ENV] = "1"
    env[LAUNCHED_BY_ENV] = sys.executable
    os.execve(str(target), [str(target), str(Path(__file__).resolve())] + sys.argv[1:], env)


def check_configuration() -> int:
    """Validate the DECLARED configuration only: no device, no subprocess, no writes.

    The recipe preflight. It must be runnable before a run directory even exists.
    """
    tf_root = Path(os.environ.get("TFNU_TF_ROOT") or E.TF_ROOT)
    report = {"tool": "verify_environment.py", "mode": "check-config",
              "tf_root": str(tf_root),
              "interpreter": sys.executable,
              "deployment_interpreter": str(E.interpreter()),
              "set_env": E.CANN_SET_ENV,
              "pythonpath": E.python_path_entries(),
              "configuration": "valid"}
    if not (tf_root / "src" / "tilefoundry").is_dir():
        report["configuration"] = "invalid"
        report["code"] = "TF_SOURCE_MISSING"
        print(json.dumps(report, indent=1, sort_keys=True))
        return 2
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PACK / ".." / ".." / ".." / "artifacts" / "verify"))
    ap.add_argument("--side", default="host")
    ap.add_argument("--device-index", type=int, default=None)
    # Defaults and their evidence live next to DEVICE_ATTEMPT_TIMEOUT_DEFAULT /
    # DEVICE_BACKOFF_DEFAULT. Env names are documented in README (seconds).
    ap.add_argument("--device-attempt-timeout-seconds", type=float,
                    default=float(os.environ.get("TFNU_DEVICE_ATTEMPT_TIMEOUT",
                                                 DEVICE_ATTEMPT_TIMEOUT_DEFAULT)),
                    help="hard bound on ONE per-card driver probe (default 20 s: a "
                         "healthy npu-smi answered in <=6 s, a D-state probe took >20 s)")
    ap.add_argument("--device-backoff-seconds", type=float,
                    default=float(os.environ.get("TFNU_DEVICE_BACKOFF",
                                                 DEVICE_BACKOFF_DEFAULT)),
                    help="wait before trying the NEXT visible card after a driver-"
                         "unresponsive probe (default 30 s; worst case 2x(20+30)=100 s)")
    ap.add_argument("--configure-cmd", default=None,
                    help="override the configure command (the mutation negative)")
    ap.add_argument("--cache-dir", default=None,
                    help="use this cache dir instead of a fresh one (the non-empty negative)")
    ap.add_argument("--negative-control", default=None,
                    choices=["skip-compile", "stale-artifact", "no-sync",
                             "empty-output", "no-readback"],
                    help="ENV-05 bad-control: inject exactly ONE fault. The run must "
                         "end non-PASS with a NAMED code; the clean run stays PASS.")
    ap.add_argument("--reuse-from", default=None,
                    help="with --negative-control stale-artifact: a directory holding a "
                         "PREVIOUS run's kernel.lib, to prove a reused binary is refused")
    ap.add_argument("--inject-error", type=int, choices=[0, 1], default=0,
                    help="1 = flip one bit of the REFERENCE, to prove the compare can fail")
    ap.add_argument("--check-config", action="store_true",
                    help="validate the declared configuration ONLY; touches no device, "
                         "starts nothing and writes nothing (the recipe preflight)")
    ap.add_argument("--skip-device", action="store_true")
    ap.add_argument("--tf-cli-cmd", default=None,
                    help="override the interpreter/command used for the TileFoundry CPU "
                         "chain; the /bin/true negative passes a command that exits 0 and "
                         "produces nothing")
    ap.add_argument("--tf-cpu-timeout", type=int, default=180,
                    help="hard bound on ONE step of checked/analyze/finalize; expiry is a "
                         "named TF_*_TIMEOUT, never an outer kill")
    ap.add_argument("--configure-timeout", type=int, default=300,
                    help="hard bound on ONE configure --findings run; expiry is the "
                         "named code CONFIGURE_TIMEOUT, never an outer kill")
    ap.add_argument("--phase-timeout", type=int, default=600,
                    help="hard bound on one phase; expiry is PHASE_TIMEOUT")
    ap.add_argument("--watchdog", type=int, default=0,
                    help="seconds after which a stack dump is printed (diagnostics only)")
    args = ap.parse_args(argv)
    # TEST SEAMS ARE NOT PRODUCTION. The probe override (and every --negative-control)
    # exists to demonstrate a refusal; it must be requested explicitly, recorded in
    # the evidence, and must never make the run count as ENV evidence.
    seam_requested = bool((os.environ.get("TFNU_PROBE_OVERRIDE") or "").strip())
    seam_allowed = (os.environ.get("TFNU_ALLOW_TEST_SEAMS") or "").strip() == "1"
    test_seam_used = seam_requested or bool(args.negative_control)
    if test_seam_used and not seam_allowed:
        print(json.dumps({
            "tool": "verify_environment.py", "verdict": "CONFIG_ERROR",
            "code": "TEST_SEAM_NOT_ALLOWED",
            "detail": "TFNU_PROBE_OVERRIDE / --negative-control require "
                      "TFNU_ALLOW_TEST_SEAMS=1; a test seam must never be active by "
                      "accident, and its run is not ENV evidence"}, indent=1))
        return EXIT_CONFIG
    if args.check_config:
        return check_configuration()
    ensure_deployment_interpreter()
    if args.watchdog:
        faulthandler.dump_traceback_later(args.watchdog, repeat=False)

    started = time.strftime("%Y%m%dT%H%M%S")
    out_root = Path(args.out)
    run_dir = out_root / ("%s_%d_%s" % (started, os.getpid(), args.side))
    run_dir.mkdir(parents=True, exist_ok=False)

    report = {"tool": "verify_environment.py", "run_dir": str(run_dir), "side": args.side,
              "started_at": started, "python": sys.version.split()[0],
              "executable": sys.executable,
              "launched_by": os.environ.get(LAUNCHED_BY_ENV),
              "test_seam_used": test_seam_used,
              "phases": [], "mutations": [], "npu_smi": {}}

    report["npu_smi"]["before"] = smi("before", run_dir)
    smi_after_done = {"flag": False}

    def finish(exit_code, verdict, code=None, detail=None):
        # The after snapshot happens EXACTLY once, whatever path got us here.
        if not smi_after_done["flag"]:
            report["npu_smi"]["after"] = smi("after", run_dir)
            smi_after_done["flag"] = True
        if report.get("test_seam_used") and verdict == "PASS":
            # The observations may be real, but a seam-controlled run is NOT evidence
            # for the ENV gate: it must carry a conclusion different from PASS.
            verdict = "PASS_TEST_SEAM_NOT_EVIDENCE"
            code = "TEST_SEAM_NOT_EVIDENCE"
            detail = ("a test seam was active; observations are shown but this run "
                      "must not be cited as ENV evidence")
            exit_code = EXIT_FAIL
        report["verdict"] = verdict
        report["code"] = code
        report["detail"] = detail
        report["finished_at"] = time.strftime("%Y%m%dT%H%M%S")
        report["phases"] = [phase.as_dict() for phase in phases]
        report["unreaped_children"] = list(UNREAPED_CHILDREN)
        evidence_path = run_dir / "evidence.json"
        evidence_path.write_text(json.dumps(report, indent=1, sort_keys=True))
        # Fsync the FILE and the DIRECTORY before exiting: os._exit skips interpreter
        # teardown, and a machine crash must not leave the only record in page cache.
        with open(evidence_path, "rb") as handle:
            os.fsync(handle.fileno())
        directory_fd = os.open(str(run_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "code": code,
                          "detail": detail, "json": str(evidence_path)}, indent=1))
        # The evidence is on disk. Measured: after a device run the interpreter's
        # teardown (torch_npu destructors) can block far outside any phase bound --
        # one run wrote PASS evidence and then sat 480+ s in shutdown. Exiting here
        # makes process lifetime as bounded as the phases it reports.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    # ---------------------------------------------------------------- phases
    phases = []

    def phase_cann():
        home = os.environ.get("ASCEND_HOME_PATH")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        facts = {"ascend_home_path": home,
                 "ascend_home_exists": bool(home) and Path(home).is_dir(),
                 "ld_library_path_has_cann": any("cann" in part for part in ld.split(os.pathsep)),
                 "set_env": CANONICAL_SET_ENV,
                 "set_env_exists": Path(CANONICAL_SET_ENV).exists()}
        if not (facts["ascend_home_exists"] and facts["ld_library_path_has_cann"]):
            return "BLOCKED", "CANN_NOT_SOURCED", \
                "source %s first: torch_npu cannot dlopen libhccl.so without it" % CANONICAL_SET_ENV, facts, EXIT_BLOCKED
        return "PASS", None, None, facts, EXIT_PASS

    def phase_layers():
        facts = {}
        try:
            import tilelang
        except Exception as exc:  # noqa: BLE001
            return "FAIL", "TILELANG_IMPORT_FAILED", "%s: %s" % (type(exc).__name__, exc), facts, EXIT_FAIL
        origin = getattr(tilelang, "__spec__", None) and tilelang.__spec__.origin
        resolved = os.path.realpath(origin or "")
        facts["origin"] = origin
        facts["version"] = getattr(tilelang, "__version__", None)
        prefix = os.path.realpath(os.environ.get("TFNU_EXPECTED_PREFIX")
                                  or os.environ.get("TFNU_PREFIX")
                                  or "/home/s00653124/workspace/envs/tfnpu-v2")
        facts["expected_prefix"] = prefix
        facts["in_prefix"] = resolved.startswith(prefix + os.sep)
        facts["l1_real_init"] = bool(origin) and origin.endswith("__init__.py")
        if not facts["l1_real_init"]:
            return "FAIL", "TILELANG_NOT_A_PACKAGE", "origin=%r" % (origin,), facts, EXIT_FAIL
        if not facts["in_prefix"]:
            return "FAIL", "TILELANG_NOT_FROM_PREFIX", \
                "resolved to %r, outside %r (a source checkout on PYTHONPATH?)" % (resolved, prefix), facts, EXIT_FAIL
        expected = os.environ.get("TFNU_EXPECTED_TILELANG_VERSION")
        facts["expected_version"] = expected
        if expected and facts["version"] != expected:
            return "FAIL", "TILELANG_WHEEL_VERSION_MISMATCH", \
                "expected %r, got %r" % (expected, facts["version"]), facts, EXIT_FAIL
        try:
            from tilelang import language as T
            facts["L3_kernel"] = hasattr(T, "Kernel")
            import tilelang.ascend
            facts["L3_ascend"] = getattr(tilelang.ascend, "__file__", None) is not None
        except Exception as exc:  # noqa: BLE001
            return "FAIL", "TILELANG_LANGUAGE_LAYER_FAILED", str(exc), facts, EXIT_FAIL
        try:
            import importlib.metadata as md
            import torch, torch_npu, tvm_ffi
            facts["tvm_ffi"] = md.version("apache-tvm-ffi")
            facts["torch"] = torch.__version__
            facts["torch_npu"] = torch_npu.__version__
        except Exception as exc:  # noqa: BLE001
            return "FAIL", "VERSION_TRIPLE_UNREADABLE", str(exc), facts, EXIT_FAIL
        if not str(facts["tvm_ffi"]).startswith("0.1.12"):
            return "FAIL", "TVM_FFI_MISMATCH", "tvm_ffi=%r" % (facts["tvm_ffi"],), facts, EXIT_FAIL
        if not (facts["L3_kernel"] and facts["L3_ascend"]):
            return "FAIL", "TILELANG_CAPABILITY_MISSING", "T.Kernel/tilelang.ascend missing", facts, EXIT_FAIL
        return "PASS", None, None, facts, EXIT_PASS

    def phase_configure():
        cmd = args.configure_cmd or "%s %s" % (sys.executable, HERE / "configure.py")
        facts = {"command": cmd}
        before = distribution_versions()
        outputs = []
        for attempt in (1, 2):
            started = time.time()
            bounded = bounded_run(cmd.split() + ["--findings"], timeout=args.configure_timeout)
            if bounded["timed_out"]:
                # Named, bounded, attributed: an outer kill would say nothing about
                # WHICH configure run hung, for how long, or whether it was even reaped.
                return "FAIL", "CONFIGURE_TIMEOUT", \
                    "attempt %d exceeded %ss (command=%r, unreaped=%s)" % (
                        attempt, args.configure_timeout, cmd.split(), bounded["unreaped"]), \
                    {"comment": "bounded by --configure-timeout", "attempt": attempt,
                     "timeout_s": args.configure_timeout, "unreaped": bounded["unreaped"],
                     "elapsed_s": round(time.time() - started, 2)}, \
                    EXIT_FAIL
            outputs.append({"attempt": attempt, "rc": bounded["rc"],
                            "elapsed_s": round(time.time() - started, 2),
                            "stdout_sha256": sha256_bytes(bounded["out"].encode()),
                            "stderr_tail": bounded["err"].strip()[-300:]})
        facts["runs"] = outputs
        facts["identical_output"] = (outputs[0]["stdout_sha256"] == outputs[1]["stdout_sha256"])
        after = distribution_versions()
        changed = {name: (before.get(name), after.get(name))
                   for name in set(before) | set(after) if before.get(name) != after.get(name)}
        facts["environment_changed"] = changed
        if changed:
            return "FAIL", "FINDINGS_MUTATED_ENV", \
                "the check wrote to the environment: %r" % (changed,), facts, EXIT_FAIL
        if any(run["rc"] != 0 for run in outputs):
            return "FAIL", "CONFIGURE_FINDINGS_FAILED", \
                "rc=%r" % ([run["rc"] for run in outputs],), facts, EXIT_FAIL
        if not facts["identical_output"]:
            return "FAIL", "CONFIGURE_NOT_IDEMPOTENT", "two runs differ", facts, EXIT_FAIL
        return "PASS", None, None, facts, EXIT_PASS

    def phase_tf_cpu():
        """ENV-04: a tiny CHECKED program through checked -> analyze -> finalize.

        No device is touched. What makes this evidence rather than a smoke test:
        the frontend must ACCEPT the program (structured result), analyze must
        produce a report naming what it executed, and finalize must write typed TIR
        that is read back and hashed. An exit 0 with no artefact fails.
        """
        tf_root = Path(os.environ.get("TFNU_TF_ROOT") or E.TF_ROOT)
        facts = {"tf_root": str(tf_root), "interpreter": sys.executable,
                 "pythonpath": E.python_path_entries(),
                 "program_source_sha256": sha256_bytes(TINY_CHECKED_PROGRAM.encode())}
        if not (tf_root / "src" / "tilefoundry").is_dir():
            return ("FAIL", "TF_SOURCE_MISSING",
                    "no TileFoundry source at %s; a checked program cannot be run "
                    "against the deployed environment" % tf_root,
                    facts, EXIT_FAIL)
        workdir = run_dir / "tf_cpu_chain"
        workdir.mkdir(parents=True, exist_ok=True)
        program = workdir / "checked_program.py"
        program.write_text(TINY_CHECKED_PROGRAM)
        env = E.runtime_env(base=dict(os.environ))
        tf_cmd = (shlex.split(args.tf_cli_cmd) if args.tf_cli_cmd
                  else [sys.executable])
        result = run_tf_cpu_chain(tf_cmd, program, workdir, env,
                                  timeout=args.tf_cpu_timeout)
        facts.update(result["facts"])
        if not result["ok"]:
            return "FAIL", result["code"], result["detail"], facts, EXIT_FAIL
        return "PASS", None, None, facts, EXIT_PASS

    def phase_compile():
        import tempfile

        import tilelang

        if args.cache_dir:
            cache = Path(args.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
        else:
            cache = Path(tempfile.mkdtemp(prefix="verify_cache_", dir=str(run_dir)))
        before = sorted(p.name for p in cache.rglob("*"))
        facts = {"cache_dir": str(cache), "cache_entries_before": before,
                 "negative_control": args.negative_control}
        # ---- ENV-05 bad controls: each injects ONE fault and must be NON-PASS.
        if args.negative_control == "skip-compile":
            facts["compile_skipped"] = True
            return ("FAIL", "COMPILE_SKIPPED",
                    "compile was deliberately skipped: an absent or stale artefact "
                    "must never be accepted as a build", facts, EXIT_FAIL)
        if args.negative_control == "stale-artifact":
            source = Path(args.reuse_from or "")
            seeded = run_dir / "compile_artifacts" / "kernel.lib"
            seeded.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                shutil.copy2(source, seeded)
                facts["seeded_stale_artifact"] = {
                    "from": str(source), "path": str(seeded),
                    "bytes": seeded.stat().st_size,
                    "sha256": sha256_bytes(seeded.read_bytes())}
            else:
                facts["seeded_stale_artifact"] = {
                    "from": str(source), "error": "no kernel.lib at --reuse-from"}
            return ("FAIL", "STALE_ARTIFACT_REUSED",
                    "a pre-existing binary was offered instead of this run's compile; "
                    "reuse is not a build", facts, EXIT_FAIL)
        if before:
            return "FAIL", "CACHE_NOT_EMPTY", \
                "a cache hit could fake a compile: %r" % (before[:5],), facts, EXIT_FAIL
        os.environ["TILELANG_CACHE_DIR"] = str(cache)
        facts["cache_enabled_before_disable"] = bool(tilelang.is_cache_enabled())
        tilelang.disable_cache()
        # EVIDENCE, not a flag we set ourselves: ask the library whether the cache
        # is still enabled after the call. A field saying "disabled" would prove
        # only that we assigned it.
        facts["cache_enabled_after_disable"] = bool(tilelang.is_cache_enabled())
        facts["disable_cache_invoked"] = True
        started_compile = time.time()
        kernel = tilelang.compile(vadd_one(N_LANES), target="ascend", out_idx=-1)
        facts["compile_seconds"] = round(time.time() - started_compile, 2)
        source = getattr(kernel, "kernel_source", "") or ""
        facts["kernel_source_chars"] = len(source)
        facts["kernel_source_sha256"] = sha256_bytes(source.encode())
        facts["adapter"] = type(getattr(kernel, "adapter", None)).__name__
        record_kernel_artifacts(kernel, run_dir / "compile_artifacts", facts)
        # FILES only: tilelang creates empty staging DIRECTORIES even when the cache
        # is disabled, and counting those as "a cache write" would fail a correct
        # run. A cached kernel is a FILE, and that is what must not exist.
        cache_after = sorted(str(p.relative_to(cache)) for p in cache.rglob("*")
                             if p.is_file())
        facts["cache_entries_after"] = cache_after
        facts["cache_remained_empty"] = not cache_after
        if not source:
            # FINISHED is not PASS: a compile that produced no source proves nothing.
            return "FAIL", "COMPILE_PRODUCED_NO_SOURCE", "empty kernel source", facts, EXIT_FAIL
        if facts["cache_enabled_after_disable"]:
            return "FAIL", "CACHE_NOT_DISABLED",                 "tilelang.is_cache_enabled() is still True after disable_cache()", facts, EXIT_FAIL
        if cache_after:
            return "FAIL", "COMPILE_WROTE_CACHE",                 "a disabled cache still received %d entries: %r" % (len(cache_after), sorted(cache_after)[:5]), facts, EXIT_FAIL
        for label in ("ascend_c_source", "linked_library"):
            if not (facts["compiled_artifacts"].get(label) or {}).get("sha256"):
                return "FAIL", "ARTIFACT_EXPORT_FAILED",                     "no %s artefact was produced: %r" % (label, facts["compiled_artifacts"].get(label)), facts, EXIT_FAIL
        report["kernel_source_sha256"] = facts["kernel_source_sha256"]
        return "PASS", None, None, facts, EXIT_PASS

    def phase_device():
        # Per-card, bounded, with a declared backoff. ONLY a driver-unresponsive
        # probe is eligible for failover; a card that is merely occupied by another
        # tenant is normal shared use and is never a reason to retry or switch.
        visible = [int(x) for x in
                   str(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or "").split(",")
                   if x.strip() != ""]
        if not visible:
            return "BLOCKED", "DEVICE_ID_MISSING", \
                "ASCEND_RT_VISIBLE_DEVICES is empty", {}, EXIT_BLOCKED
        if args.device_index is not None and not 0 <= args.device_index < len(visible):
            return "BLOCKED", "DEVICE_INDEX_NOT_VISIBLE", \
                "visible=%r requested=%r" % (visible, args.device_index), \
                {"visible_list": visible}, EXIT_BLOCKED
        candidates = (list(range(args.device_index, len(visible)))
                      if args.device_index is not None else list(range(len(visible))))
        attempt_timeout = args.device_attempt_timeout_seconds
        backoff = args.device_backoff_seconds
        facts = {"visible_list": visible, "candidates": candidates,
                 "attempt_timeout_seconds": attempt_timeout,
                 "backoff_seconds": backoff,
                 "attempt_timeout_basis": (
                     "healthy npu-smi measured 0.77-1.4 s, worst healthy 6 s; D-state "
                     "measured >20 s and unreclaimable after SIGKILL"),
                 "backoff_basis": (
                     "one window for transient congestion to clear; attempts are "
                     "bounded by the visible list, worst case 2x(20+30)=100 s")}
        attempts, selected, failure = first_responsive_card(
            candidates, visible,
            lambda physical: probe_card(physical, attempt_timeout), backoff)
        facts["device_attempts"] = attempts
        facts["all_cards_unresponsive"] = selected is None and failure is None
        if failure is not None:
            # A named probe failure (fast rc!=0, empty output, unattributed timeout)
            # is NOT driver unresponsiveness: no backoff was waiting, no later card
            # was probed, and this run is not a PASS.
            facts["device_probe_failure"] = failure
            return ("BLOCKED", failure["code"],
                    "device probe failed on physical id %s: %s (rc=%r timed_out=%r "
                    "unreaped=%r wchan=%r stderr=%r); no backoff, no failover, not PASS"
                    % (failure["physical_device_id"], failure["code"], failure["rc"],
                       failure["timed_out"], failure["unreaped"], failure["wchan"],
                       failure["stderr_tail"]), facts, EXIT_BLOCKED)
        if selected is None:
            return ("NOT_RUN", BUSY_CODE,
                    "every visible card was driver-unresponsive: %d bounded attempt(s), "
                    "attempt_timeout=%ss, backoff=%ss; all_cards_unresponsive=true"
                    % (len(attempts), attempt_timeout, backoff), facts, EXIT_NOT_RUN)
        facts["selected_device"] = selected
        facts.update(selected)
        facts["lock_path"] = str(TEAM_LOCK)
        facts["lock_canonical"] = TEAM_LOCK == TEAM_LOCK_PATH
        if os.environ.get("TF_DEVICE_LOCK_HELD", "").strip() == "1":
            facts["lock"] = "held by parent"
        else:
            try:
                handle = acquire_team_lock(TEAM_LOCK)
            except BlockingIOError:
                return "BLOCKED", "TEAM_LOCK_HELD", \
                    "another team measurement job holds %s" % TEAM_LOCK, facts, EXIT_BLOCKED
            except OSError as exc:
                return "BLOCKED", "TEAM_LOCK_UNOPENABLE", str(exc), facts, EXIT_BLOCKED
            facts["lock"] = "acquired"
            report.setdefault("locks", []).append(str(TEAM_LOCK))

        import numpy as np
        import torch
        import torch_npu  # noqa: F401
        import tilelang

        logical_index = int(facts["logical_index"])
        torch.npu.set_device(logical_index)
        facts["device_name"] = torch.npu.get_device_name(logical_index)
        started_compile = time.time()
        kernel = tilelang.compile(vadd_one(N_LANES), target="ascend", out_idx=-1)
        facts["device_compile_seconds"] = round(time.time() - started_compile, 2)
        record_kernel_artifacts(kernel, run_dir / "device_artifacts", facts)
        host_in = np.arange(N_LANES, dtype=np.float32) + 0.5
        expected = (host_in + np.float32(1.0)).astype(np.float32)
        if args.inject_error:
            # Disturb the REFERENCE by one bit: if the comparison still passes, it
            # cannot detect anything and the whole device phase is theatre.
            expected = expected.copy()
            expected[7] = np.nextafter(expected[7], np.float32(1e9))
            facts["injected_error"] = "one ulp on reference lane 7"
        tensor = torch.from_numpy(host_in.copy()).to("npu:%d" % logical_index)
        facts["launch"] = "kernel(tensor) on npu:%d" % logical_index
        produced = kernel(tensor)
        if args.negative_control == "no-sync":
            # ENV-05 control: launch and read WITHOUT synchronize. The run must not
            # be able to pass, because an unsynchronised read is not a result.
            facts["synchronize_called"] = False
        else:
            torch.npu.synchronize()
            facts["synchronize_called"] = True
        if args.negative_control == "no-readback":
            facts["readback_source"] = "SKIPPED (negative control)"
            return ("FAIL", "READBACK_SKIPPED",
                    "the output was not read back, so no element was compared",
                    facts, EXIT_FAIL)
        if args.negative_control == "empty-output":
            got = np.array([], dtype=np.float32)
            facts["readback_source"] = "EMPTY (negative control)"
        else:
            got = produced.cpu().numpy()
        facts["elements_expected"] = int(N_LANES)
        facts["input_elements"] = int(host_in.size)
        facts["elements_read"] = int(got.size)
        facts["input_nonempty"] = bool(host_in.size)
        facts["all_elements_compared"] = bool(
            host_in.size == got.size == expected.size == N_LANES)
        facts.setdefault(
            "readback_source", "host numpy copy after torch.npu.synchronize()")
        if not facts.get("synchronize_called"):
            return ("FAIL", "SYNCHRONIZE_SKIPPED",
                    "launch returned without torch.npu.synchronize(); the readback is "
                    "not an observed result", facts, EXIT_FAIL)
        if not facts["all_elements_compared"] or host_in.size == 0:
            return "FAIL", "READBACK_INCOMPLETE", \
                ("readback is not a full comparison: expected=%d input=%d read=%d"
                 % (getattr(expected, "size", -1), host_in.size, got.size)), facts, EXIT_FAIL
        facts["input_sha256"] = sha256_bytes(host_in.tobytes())
        facts["output_sha256"] = sha256_bytes(got.tobytes())
        facts["expected_sha256"] = sha256_bytes(expected.tobytes())
        bad = np.flatnonzero(got != expected)
        facts["mismatch_count"] = int(bad.size)
        facts["mismatch_indices"] = bad[:8].tolist()
        if bad.size:
            return "FAIL", "VADD_MISMATCH", \
                "%d lanes differ from the host reference" % int(bad.size), facts, EXIT_FAIL
        for label in ("ascend_c_source", "linked_library"):
            if not (facts["compiled_artifacts"].get(label) or {}).get("sha256"):
                return "FAIL", "DEVICE_ARTIFACT_EXPORT_FAILED", \
                    "no device %s artefact was produced: %r" % (label, facts["compiled_artifacts"].get(label)), facts, EXIT_FAIL
        return "PASS", None, None, facts, EXIT_PASS

    for pid, deps, fn in (("cann", [], phase_cann),
                          ("layers", ["cann"], phase_layers),
                          ("configure_idempotent", ["layers"], phase_configure),
                          ("tf_cpu_tools", ["configure_idempotent"], phase_tf_cpu),
                          ("compile_smoke", ["tf_cpu_tools"], phase_compile),
                          ("device_smoke", ["compile_smoke"], phase_device)):
        phases.append(Phase(pid, deps, fn))

    blocked_exit = EXIT_PASS
    for phase in phases:
        if any(other.status != "PASS" for other in phases if other.id in phase.depends_on):
            phase.status = "NOT_RUN"
            phase.detail = "a dependency did not pass"
            continue
        if phase.id == "device_smoke" and args.skip_device:
            phase.status = "NOT_RUN"
            phase.detail = "--skip-device"
            continue
        started_phase = time.time()
        print("phase start: %s" % phase.id, file=sys.stderr, flush=True)
        try:
            if args.phase_timeout:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(int(args.phase_timeout))
            try:
                status, code, detail, facts, hint = phase.fn()
            finally:
                signal.alarm(0)
        except PhaseTimeout:
            status, code, detail, facts, hint = (
                "FAIL", "PHASE_TIMEOUT",
                "phase %r exceeded %ss" % (phase.id, args.phase_timeout),
                {"timeout_s": args.phase_timeout}, EXIT_FAIL)
        except Exception as exc:  # noqa: BLE001
            status, code, detail, facts, hint = ("FAIL", "PHASE_EXCEPTION",
                                                 "%s: %s" % (type(exc).__name__, exc), {}, EXIT_FAIL)
        print("phase done: %s -> %s %s" % (phase.id, status, code or ""),
              file=sys.stderr, flush=True)
        phase.status, phase.code, phase.detail, phase.facts = status, code, detail, facts
        phase.seconds = round(time.time() - started_phase, 2)
        phase.exit_hint = hint
        if status == "BLOCKED":
            blocked_exit = EXIT_BLOCKED
        if status == "FAIL":
            break

    statuses = {phase.id: phase.status for phase in phases}
    failed = [pid for pid, st in statuses.items() if st == "FAIL"]
    blocked = [pid for pid, st in statuses.items() if st == "BLOCKED"]
    not_run = [pid for pid, st in statuses.items() if st == "NOT_RUN"]
    if failed:
        first = next(phase for phase in phases if phase.id == failed[0])
        return finish(EXIT_FAIL, "FAIL", first.code, first.detail)
    if blocked:
        first = next(phase for phase in phases if phase.id == blocked[0])
        return finish(EXIT_BLOCKED, "BLOCKED", first.code, first.detail)
    if not_run:
        # BUSY_DEVICE_NOT_RESPONDING and friends: nothing was measured on the card, so
        # this is NOT_RUN -- never PASS, and never a software FAIL.
        first = next(phase for phase in phases if phase.id == not_run[0])
        return finish(EXIT_NOT_RUN, "NOT_RUN", first.code, first.detail)
    return finish(EXIT_PASS, "PASS", None, None)


if __name__ == "__main__":
    raise SystemExit(main())
