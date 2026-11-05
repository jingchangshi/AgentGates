"""Where the shared interpreter lives and what it needs -- one source of truth.

Host and container see the SAME paths (/home is a bind mount), so there is
deliberately NO host/container branch here. A branch would be the bug: it would
let the two sides drift into "works on the host" while the container silently
uses something else.

This module answers three questions and nothing else:
  1. which interpreter is the deployment interpreter (not "any python"),
  2. what environment it must be given (four pins),
  3. whether it is actually usable (four verify layers, never a name check).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: The workspace that host and container share. Overridable only for tests and
#: relocation; the default IS the deployment.
WORKSPACE = Path(os.environ.get("TFNU_WORKSPACE") or "/home/s00653124/workspace")

#: The shared prefix. A venv, so the interpreter path is part of the contract.
#: DEFAULT IS v2: tilelang built from main@0e687c17. v1 is the feat/hivm branch and
#: is kept ONLY as a rollback/control environment -- it must be selected explicitly
#: (TFNU_PREFIX=.../envs/tfnpu-v1), and that selection is reported as
#: env.rollback_prefix=true so a run can never quietly still be the old wheel.
SHARED_PREFIX = Path(os.environ.get("TFNU_PREFIX")
                     or (WORKSPACE / "envs" / "tfnpu-v2"))

#: The rollback/control prefix. Selecting it is legitimate, but it must be visible.
ROLLBACK_PREFIX = WORKSPACE / "envs" / "tfnpu-v1"

#: Every wheel ever installed here, kept so a rebuild needs no network.
#: TFNU_WHEELS points at a different cache (a rebuilt wheel lives in its own dir).
WHEEL_CACHE = Path(os.environ.get("TFNU_WHEELS") or (WORKSPACE / "envs" / "wheels"))

TILELANG_ROOT = WORKSPACE / "tilelang"
TF_ROOT = WORKSPACE / "TileFoundry"

#: CANN must be sourced before torch_npu is imported, or it fails on libhccl.so.
CANN_SET_ENV = os.environ.get(
    "TFNU_CANN_SET_ENV", "/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh"
)

#: The tilelang source revision the wheel is built from. A wheel built from an
#: unknown tree is not reproducible, so the commit is recorded and checked.
TILELANG_COMMIT = "0e687c17"

#: The feat/hivm revision the v1 ROLLBACK prefix was built from. Recorded so a
#: control run can say which tree it came from; never used as the default.
TILELANG_COMMIT_ROLLBACK = "8e2d3c82"

#: The version string the INSTALLED wheel must report. It is not cosmetic: the
#: wheel and the source tree disagree on the tag by construction -- the wheel is
#: +cpu (built CPU-only) while the same commit in the tree reports +cuda. A probe
#: that accepts either cannot tell "the wheel is installed" from "someone put the
#: tree on PYTHONPATH", which are completely different deployments.
TILELANG_WHEEL_VERSION = "0.1.15+cpu.git0e687c17"


def prefix_is_rollback() -> bool:
    """True when the selected prefix is the feat-branch rollback environment.

    Reported in configure facts as env.rollback_prefix. A control environment is
    allowed; an UNLABELLED one is the defect this exists to prevent.
    """
    try:
        return Path(SHARED_PREFIX).resolve() == Path(ROLLBACK_PREFIX).resolve()
    except OSError:
        return False

#: Exact pins. tvm_ffi is pinned to 0.1.12 for a specific reason: a newer tvm_ffi
#: (0.1.15.post2 was measured) fails with
#:   RuntimeError: 'tl.SwizzleMode' has no enum at string index 'NONE'
#: which looks like a broken tilelang install but is only a version skew. The
#: failure names no version, which is exactly why it is pinned here by name.
#:
#: z3-solver is not a free choice either: the built wheel links libtvm_compiler.so
#: against a SONAME that fixes it. Measured -- the wheel was built in an isolated
#: env that honoured tilelang's own requirement "z3-solver<4.15.5,>=4.13.0", so it
#: links libz3.so.4.15, while a 5.3.x z3-solver ships only libz3.so.5.3. The result
#:   OSError: libz3.so.4.15: cannot open shared object file
#: appears at "import tilelang" and reads like a broken install. 4.15.4.0 is the
#: newest release inside tilelang's range; the intersection is ASSERTED at
#: provision time by z3_requirement(), so the skew cannot come back silently.
PINS = {
    "torch": "2.12.0+cpu",
    "torch_npu": "2.12.0",
    "apache-tvm-ffi": "0.1.12",
    "numpy": "2.5.4",
    "cloudpickle": "3.1.2",
    "psutil": "7.2.2",
    "tqdm": "4.70.1",
    "typing_extensions": "4.16.0",
    "z3-solver": "4.15.4.0",
}

#: tilelang's own z3 range, recorded as the FLOOR of the intersection. The source
#: of truth is the installed wheel's METADATA (z3_requirement() re-reads it), so a
#: rebuilt wheel with a different range is caught instead of being assumed.
TILELANG_Z3_REQUIREMENT = "z3-solver<4.15.5,>=4.13.0"
TILELANG_Z3_REQUIREMENT_SOURCE = (
    "tilelang pyproject.toml:44 / installed tilelang METADATA Requires-Dist"
)

#: Everything torch_npu/tilelang import at runtime. Measured by diffing against a
#: known-good environment rather than guessed: omitting one produces a
#: ModuleNotFoundError deep inside torch_npu (yaml was the first one hit).
RUNTIME_PINS = {
    "absl-py": "2.5.1",
    "attrs": "26.1.0",
    "decorator": "5.3.1",
    "graphviz": "0.21",
    "immutabledict": "4.3.1",
    "isl-python": "0.1.11",
    "ortools": "9.15.6755",
    "packaging": "26.3",
    "pandas": "3.1.0",
    "protobuf": "6.33.6",
    "python-dateutil": "4.0.0",
    "PyYAML": "6.0.3",
    "tornado": "6.6",
    "tzdata": "2026.6",
    "wheel": "0.48.0",
}

#: The whole set, for inspection and for planning an install.
ALL_PINS = {**PINS, **RUNTIME_PINS}


def interpreter() -> Path:
    """The deployment interpreter. Not "python", not "the one on PATH"."""
    return SHARED_PREFIX / "bin" / "python"


def python_path_entries() -> list[str]:
    """Entries the shared prefix must see. tilelang FIRST so its own tree wins."""
    return [
        str(TILELANG_ROOT),
        str(TF_ROOT / "src"),
        str(TF_ROOT / "targets" / "ascend"),
    ]


def runtime_env(base: dict | None = None, *, with_cann: bool = True) -> dict:
    """The environment the deployment interpreter must be given.

    Four pins, each with a reason:
      - PYTHONPATH carries the tilelang TREE: the wheel installs the package, but
        the tree entry is what makes the pinned revision win over a stale copy.
      - TORCH_DEVICE_BACKEND_AUTOLOAD=0 is REQUIRED here: with it on, torch's own
        autoload raises on torch_npu before this process can say anything useful.
      - CANN sourced, because torch_npu imports libhccl.so at import time.
      - LD_LIBRARY_PATH is deliberately NOT touched for tilelang: tilelang finds
        its own libs through TL_LIBS (tilelang/env.py), so writing LD_LIBRARY_PATH
        would look like work while changing nothing.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries())
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env.pop("LD_LIBRARY_PATH_FOR_TILELANG", None)
    if with_cann:
        env["TFNU_SOURCE_CANN"] = CANN_SET_ENV
    return env


#: The verify probe. Runs INSIDE the target interpreter and prints one JSON object.
#: Four layers, because a name check is not a capability check: a namespace package
#: satisfies find_spec() while providing nothing (measured: origin=None from a bare
#: `tilelang/` directory on sys.path).
PROBE = r'''
import json, os, sys
out = {"interpreter": sys.executable, "python": sys.version.split()[0]}
# CANN is a PRECONDITION of every import below, so it is checked FIRST. Without
# it, torch_npu cannot dlopen libhccl.so and torch's autoload raises a message
# that blames the backend extension -- a symptom that reads like "tilelang is
# broken". Naming the cause here is the whole point of this block.
out["cann"] = {
    "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
    "ascend_toolkit_home": os.environ.get("ASCEND_TOOLKIT_HOME"),
    "ld_library_path_has_cann": any(
        "cann" in e for e in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    ),
    "rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
}
out["cann"]["sourced"] = bool(
    out["cann"]["ascend_home_path"] and out["cann"]["ld_library_path_has_cann"]
)
if not out["cann"]["sourced"]:
    out["ok"] = False
    out["code"] = "CANN_NOT_SOURCED"
    out["error"] = "CANN_NOT_SOURCED: ASCEND_HOME_PATH unset or LD_LIBRARY_PATH has no CANN entry"
    print(json.dumps(out))
    raise SystemExit(3)
try:
    import importlib.util as u
    spec = u.find_spec("tilelang")
    out["L1_found"] = spec is not None
    origin = getattr(spec, "origin", None) if spec else None
    out["L1_origin"] = origin
    # L1: a namespace package has origin None; it is present in name only.
    out["L1_real_package"] = bool(origin) and origin.endswith("__init__.py")
    if not out["L1_real_package"]:
        raise RuntimeError(
            "tilelang is not a real package (origin=%r); a namespace package "
            "satisfies find_spec but provides nothing" % (origin,)
        )
    import tilelang
    out["L2_imported"] = True
    out["tilelang"] = getattr(tilelang, "__version__", None)
    out["L2_file"] = getattr(tilelang, "__file__", None)
    # The artefact under test must be the WHEEL IN THE SHARED PREFIX. Resolving to
    # the source tree means the check inspected something nobody deploys.
    expected_prefix = os.environ.get("TFNU_EXPECTED_PREFIX")
    expected_version = os.environ.get("TFNU_EXPECTED_TILELANG_VERSION")
    resolved = os.path.realpath(out["L2_file"] or "")
    expected_root = os.path.realpath(expected_prefix or "") + os.sep
    out["L2_resolved"] = resolved
    out["L2_expected_prefix"] = expected_prefix
    out["L2_in_shared_prefix"] = bool(expected_prefix) and resolved.startswith(expected_root)
    if not out["L2_in_shared_prefix"]:
        out["ok"] = False
        out["code"] = "TILELANG_RESOLVES_TO_SOURCE_TREE"
        out["error"] = (
            "TILELANG_RESOLVES_TO_SOURCE_TREE: tilelang resolved to %r, which is not "
            "under %r; the source tree is on the import path, so this probe would "
            "say nothing about the installed wheel" % (resolved, expected_prefix)
        )
        print(json.dumps(out))
        raise SystemExit(4)
    if expected_version and out["tilelang"] != expected_version:
        out["ok"] = False
        out["code"] = "TILELANG_WHEEL_VERSION_MISMATCH"
        out["error"] = "expected %r, got %r" % (expected_version, out["tilelang"])
        print(json.dumps(out))
        raise SystemExit(5)
    from tilelang import language as T
    # L3: the two capabilities a usable install must have.
    out["L3_kernel"] = hasattr(T, "Kernel")
    import tilelang.ascend
    out["L3_ascend"] = tilelang.ascend.__name__
    import torch, torch_npu, tvm_ffi
    out["L4_torch"] = torch.__version__
    out["L4_torch_npu"] = torch_npu.__version__
    out["L4_tvm_ffi"] = getattr(tvm_ffi, "__version__", None)
    out["npapi_available"] = bool(torch.npu.is_available())
    out["npu_count"] = int(torch.npu.device_count())
    out["ok"] = True
except Exception as error:
    out["ok"] = False
    out["error"] = "%s: %s" % (type(error).__name__, error)
print(json.dumps(out))
'''


def probe_env(base: dict | None = None) -> dict:
    """Environment for the VERIFY probe: the wheel, never the source tree.

    python_path_entries() puts the tilelang TREE first on purpose for TileFoundry
    runs, but reusing that here was a real defect: the probe then reported
    "0.1.15+cuda.git8e2d3c82" from workspace/tilelang and passed, while saying
    nothing about the wheel that a real run imports. An environment check that
    inspects a different artefact than the one deployed is a vacuous pass.
    """
    env = runtime_env(base, with_cann=False)
    env["PYTHONPATH"] = os.pathsep.join(
        [entry for entry in python_path_entries() if entry != str(TILELANG_ROOT)]
    )
    env["TFNU_EXPECTED_PREFIX"] = str(SHARED_PREFIX)
    # The expected version follows the SELECTED prefix: a rebuilt wheel (main 0e687c17
    # vs feat 8e2d3c82) reports a different git suffix, and hard-coding one would fail
    # the other by design.
    env["TFNU_EXPECTED_TILELANG_VERSION"] = (
        os.environ.get("TFNU_EXPECTED_TILELANG_VERSION") or TILELANG_WHEEL_VERSION
    )
    return env


def probe_source_file() -> Path:
    """Where PROBE is materialised. Kept OUT of the workspace on purpose."""
    return Path("/tmp") / ("tfnpu_probe_%d.py" % os.getpid())


def run_probe(*, source_cann: bool = True, timeout: int = 600) -> dict:
    """Run the four-layer probe in the deployment interpreter; return its facts."""
    target = interpreter()
    if not target.exists():
        return {"ok": False, "error": "no interpreter at %s" % target}
    path = probe_source_file()
    path.write_text(PROBE, encoding="utf-8")
    # Run from a directory that holds no modules: a stray file on sys.path would
    # shadow a stdlib name and the failure would look like a broken install.
    cwd = "/tmp"
    env = probe_env()
    command = [str(target), str(path)]
    if source_cann and Path(CANN_SET_ENV).exists():
        command = [
            "bash",
            "-lc",
            "set -e; . %s >/dev/null 2>&1; exec %s %s"
            % (CANN_SET_ENV, target, path),
        ]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "probe timed out after %ss" % timeout}
    finally:
        path.unlink(missing_ok=True)
    payload = None
    for line in reversed((done.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            break
    if payload is None:
        return {
            "ok": False,
            "error": "probe produced no JSON (rc=%s)" % done.returncode,
            "stdout_tail": (done.stdout or "")[-600:],
            "stderr_tail": (done.stderr or "")[-600:],
        }
    return payload


def _strip_cann(env: dict) -> dict:
    """The same env with every CANN trace removed -- the negative control's input."""
    clean = dict(env)
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "BISHENG_HOME",
                "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "ASCEND_RT_VISIBLE_DEVICES"):
        clean.pop(key, None)
    ld = clean.get("LD_LIBRARY_PATH", "")
    kept = [entry for entry in ld.split(os.pathsep) if entry and "cann" not in entry
            and "Ascend" not in entry and "ascend" not in entry]
    if kept:
        clean["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    else:
        clean.pop("LD_LIBRARY_PATH", None)
    return clean


def negative_probe(timeout: int = 600) -> dict:
    """Negative control: provenance of the CANN precondition.

    Without CANN, importing tilelang must fail NAMING the CANN gap
    (CANN_NOT_SOURCED), not with a bare OSError about libhccl/libz3 that invites
    the reader to blame the tilelang wheel. Asserted, not asserted-by-comment.
    """
    target = interpreter()
    if not target.exists():
        return {"ok": False, "error": "no interpreter at %s" % target}
    path = Path("/tmp") / ("tfnpu_negative_%d.py" % os.getpid())
    path.write_text(PROBE, encoding="utf-8")
    env = _strip_cann(probe_env())
    try:
        done = subprocess.run([str(target), str(path)], capture_output=True, text=True,
                              timeout=timeout, cwd="/tmp", env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "negative probe timed out after %ss" % timeout}
    finally:
        path.unlink(missing_ok=True)
    payload = None
    for line in reversed((done.stdout or "").strip().splitlines()):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                payload = json.loads(line.strip())
                break
            except ValueError:
                continue
    got = (payload or {}).get("code")
    return {
        "ok": got == "CANN_NOT_SOURCED",
        "expected_code": "CANN_NOT_SOURCED",
        "observed_code": got,
        "observed_error": (payload or {}).get("error"),
        "rc": done.returncode,
        "stderr_tail": (done.stderr or "")[-400:],
    }


def pin_problems(installed: dict) -> dict:
    """Which pins are absent or at the wrong version, and why each matters."""
    problems = {}
    for name, want in ALL_PINS.items():
        have = installed.get(name) or installed.get(name.lower()) or installed.get(
            name.replace("-", "_")
        )
        if have is None:
            problems[name] = {"want": want, "have": None}
        elif have != want:
            problems[name] = {"want": want, "have": have}
    return problems


# ---------------------------------------------------------------------------
# z3: the one pin whose VALUE is dictated by the wheel's binary, not by taste.
# ---------------------------------------------------------------------------

def _parse_release(version: str) -> tuple:
    """Numeric release tuple of a PEP 440 version, ignoring local/pre suffixes."""
    import re as _re

    head = version.split("+", 1)[0]
    release = _re.split(r"[._-]", head)
    out = []
    for part in release:
        digits = _re.match(r"^(\d+)", part)
        if not digits:
            break
        out.append(int(digits.group(1)))
    return tuple(out)


def _satisfies(version: str, specifiers: list[tuple[str, str]]) -> bool:
    have = _parse_release(version)
    for op, want in specifiers:
        ref = _parse_release(want)
        width = max(len(have), len(ref))
        left = have + (0,) * (width - len(have))
        right = ref + (0,) * (width - len(ref))
        if op == ">=" and not left >= right:
            return False
        if op == ">" and not left > right:
            return False
        if op == "<=" and not left <= right:
            return False
        if op == "<" and not left < right:
            return False
        if op == "==" and not left == right:
            return False
        if op == "!=" and not left != right:
            return False
    return True


def _specifiers_from(declared: str) -> list[tuple[str, str]]:
    """(op, version) pairs out of a requirement string, name prefix and all.

    Accepts BOTH metadata spellings: "z3-solver<4.15.5,>=4.13.0" (comma form) and
    "z3-solver (<4.15.5,>=4.13.0)" (parenthesised). Getting this wrong is not
    cosmetic: dropping the upper bound turns the check into a rubber stamp, which
    is exactly how 5.3.3.0 shipped once.
    """
    import re as _re

    return [
        (op, version)
        for op, version in _re.findall(r"(>=|<=|==|!=|>|<)\s*([0-9][0-9A-Za-z.]*)", declared)
    ]


def _wheel_z3_requirement() -> tuple[str | None, str | None]:
    """The z3 range the INSTALLED tilelang wheel declares, read from its METADATA.

    Read from the artifact rather than hard-coded: a rebuilt wheel with a different
    range must change this answer, or the check would be theatre.
    """
    patterns = ("lib/python*/site-packages/tilelang-*.dist-info/METADATA",
                "lib64/python*/site-packages/tilelang-*.dist-info/METADATA")
    for pattern in patterns:
        for metadata in sorted(SHARED_PREFIX.glob(pattern)):
            try:
                text = metadata.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if line.lower().startswith("requires-dist:") and "z3-solver" in line.lower():
                    value = line.split(":", 1)[1].strip().split(";", 1)[0].strip()
                    return value, str(metadata)
    return None, None


def z3_requirement(installed: dict | None = None) -> dict:
    """Assert: tilelang's declared z3 range INTERSECTS the installed z3 version.

    A pin that sits outside the wheel's own range is not "slightly off": it fails
    at import with an OSError naming a .so, which points at the wrong thing. This
    refuses by name at provision time instead (Z3_PIN_OUTSIDE_TILELANG_RANGE).
    """
    installed = installed or {}
    declared, source = _wheel_z3_requirement()
    if declared is None:
        declared, source = TILELANG_Z3_REQUIREMENT, TILELANG_Z3_REQUIREMENT_SOURCE
    specifiers = _specifiers_from(declared)
    want_pin = ALL_PINS.get("z3-solver")
    have = installed.get("z3-solver") or installed.get("z3_solver") or want_pin
    facts = {
        "requirement": declared,
        "requirement_source": source,
        "specifiers": specifiers,
        "pin": want_pin,
        "installed": have,
        "pin_in_range": bool(installed and _satisfies(want_pin, specifiers)),
        "installed_in_range": bool(have and _satisfies(have, specifiers)),
    }
    facts["ok"] = bool(
        specifiers and facts["installed_in_range"] and facts["pin_in_range"]
    )
    if not facts["ok"]:
        facts["code"] = "Z3_PIN_OUTSIDE_TILELANG_RANGE"
    return facts


# ---------------------------------------------------------------------------
# CANN: a PRECONDITION of the import check, not a neighbouring convenience.
# ---------------------------------------------------------------------------

def cann_state() -> dict:
    """What sourcing set_env.sh actually did to THIS process's environment.

    Order matters and is part of the contract: set_env.sh overwrites
    ASCEND_RT_VISIBLE_DEVICES, so the device pin must be exported afterwards.
    Recorded here so a run can say which script was sourced and what it resolved.
    """
    home = os.environ.get("ASCEND_HOME_PATH")
    state = {
        "set_env_script": CANN_SET_ENV,
        "set_env_exists": Path(CANN_SET_ENV).exists(),
        "ascend_home_path": home,
        "ascend_home_path_exists": bool(home) and Path(home).is_dir(),
        "ascend_toolkit_home": os.environ.get("ASCEND_TOOLKIT_HOME"),
        "ld_library_path_has_cann": any(
            "cann" in entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        ),
        "rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "bisheng_home": os.environ.get("BISHENG_HOME"),
        "bisheng_on_path": None,
    }
    import shutil as _shutil

    state["bisheng_on_path"] = _shutil.which("bisheng")
    state["sourced"] = bool(state["ascend_home_path_exists"] and state["ld_library_path_has_cann"])
    if not state["sourced"]:
        # Name the consequence, not the symptom: without CANN, torch_npu cannot
        # dlopen libhccl.so and the error surfaces as a torch backend failure.
        state["code"] = "CANN_NOT_SOURCED"
        state["consequence"] = (
            "torch_npu import fails on libhccl.so; torch's device-backend autoload "
            "then raises 'Failed to load the backend extension: torch_npu'"
        )
    elif not state["bisheng_on_path"]:
        state["code"] = "CANN_TOOLCHAIN_INCOMPLETE"
        state["consequence"] = "CANN is set but no bisheng on PATH; CCE compile would fail late"
    return state



#: The CCE compiler is NOT vendored by tilelang. tilelang/contrib/bisheng.py only
#: emits Ascend C (.asc) and then shelling out to an EXTERNAL bisheng, linking with
#: ld.lld (bisheng's sibling). Without it, import and lowering still succeed and the
#: failure only appears at "compile to a device binary", reporting the canned
#: message about installing bisheng. That is the "looks fine until the card" class
#: of defect, so the toolchain is a NAMED dependency of this profile.
TOOLCHAIN = ("bisheng", "ld.lld", "bishengir-compile")

#: Only these gate usability for the profile TileFoundry actually uses: tilelang
#: emits Ascend C/CCE through bisheng and links with its sibling ld.lld.
#: bishengir-compile belongs to the HIVM path, which TileFoundry does not use, so
#: it is RECORDED (for forensics) but never required -- a missing one is not a
#: blocker and must not masquerade as one.
REQUIRED_TOOLS = ("bisheng", "ld.lld")


def ascend_home() -> str | None:
    """Where CANN says its toolkit lives; set only after set_env.sh is sourced."""
    value = os.environ.get("ASCEND_HOME_PATH")
    return value or None


def toolchain_env(base: dict | None = None) -> dict:
    """Environment that makes the CANN toolchain resolvable WITHOUT host PATH.

    BISHENG_HOME is set to ASCEND_HOME_PATH explicitly. tilelang tries
    $BISHENG_HOME/bin/bisheng first and only then falls back to shutil.which, so
    relying on the caller's PATH would make the build depend on an ambient value
    nobody recorded -- exactly the implicit dependency this profile forbids.
    """
    env = runtime_env(base)
    home = ascend_home()
    if home:
        env["BISHENG_HOME"] = home
    return env


def probe_toolchain(timeout: int = 120) -> dict:
    """realpath + version + sha256 for each compiler, or a NAMED refusal.

    A version string alone cannot pin a compiler: two builds can share a version
    and differ. The hash is recorded so a run can say which binary it used.
    """
    import hashlib

    facts = {"tools": {}}
    home = ascend_home()
    facts["ascend_home_path"] = home
    facts["bisheng_home"] = os.environ.get("BISHENG_HOME")
    for name in TOOLCHAIN:
        resolved = None
        if name == "bisheng" and home:
            explicit = Path(home) / "bin" / "bisheng"
            if explicit.exists():
                resolved = str(explicit)
        if resolved is None:
            import shutil

            resolved = shutil.which(name)
        entry = {"which": resolved}
        if resolved is None:
            entry["error"] = "not found on PATH and BISHENG_HOME/bin has no %s" % name
            facts["tools"][name] = entry
            continue
        real = Path(resolved).resolve()
        entry["realpath"] = str(real)
        try:
            entry["sha256"] = hashlib.sha256(real.read_bytes()).hexdigest()
        except OSError as error:
            entry["sha256"] = None
            entry["hash_error"] = str(error)
        rc, out, err = _subprocess_version(real, timeout)
        entry["version_rc"] = rc
        entry["version"] = (out or err or "").strip().splitlines()[:2]
        facts["tools"][name] = entry
    for name, entry in facts["tools"].items():
        entry["required"] = name in REQUIRED_TOOLS
    missing = [n for n in REQUIRED_TOOLS if facts["tools"][n].get("which") is None]
    facts["required"] = list(REQUIRED_TOOLS)
    facts["informational"] = [n for n in TOOLCHAIN if n not in REQUIRED_TOOLS]
    facts["ok"] = not missing
    if missing:
        facts["code"] = "CANN_TOOLCHAIN_INCOMPLETE: %s" % missing[0]
        facts["missing"] = missing
    return facts


def _subprocess_version(binary: Path, timeout: int):
    try:
        done = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=timeout
        )
        return done.returncode, done.stdout, done.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, "", str(error)

__all__ = [
    "ALL_PINS",
    "TOOLCHAIN",
    "CANN_SET_ENV",
    "PINS",
    "RUNTIME_PINS",
    "SHARED_PREFIX",
    "TILELANG_COMMIT",
    "TILELANG_ROOT",
    "TILELANG_WHEEL_VERSION",
    "probe_env",
    "TF_ROOT",
    "WHEEL_CACHE",
    "WORKSPACE",
    "interpreter",
    "pin_problems",
    "python_path_entries",
    "run_probe",
    "runtime_env",
    "toolchain_env",
    "TOOLCHAIN",
    "probe_toolchain",
    "ROLLBACK_PREFIX",
    "prefix_is_rollback",
    "REQUIRED_TOOLS",
    "TILELANG_Z3_REQUIREMENT",
    "TILELANG_Z3_REQUIREMENT_SOURCE",
    "cann_state",
    "negative_probe",
    "z3_requirement",
]
