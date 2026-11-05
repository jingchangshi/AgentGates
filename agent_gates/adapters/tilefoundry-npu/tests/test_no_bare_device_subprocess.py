"""No device-touching module may call subprocess.run(timeout=...) directly.

Measured defect this pins: subprocess.run(timeout=N) kills the child and then WAITS for
it. A child blocked in uninterruptible sleep (D) ignores SIGKILL, so the wait never
returns and the "timeout" is unbounded. Every device call must go through bounded_run(),
which uses its own process group, kills the GROUP, and abandons an unreaped child with
unreaped=True instead of blocking.
"""
from __future__ import annotations

import ast
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"

#: Modules that touch the device (or launch children that do).
#: verify_environment.py is the module that launches device probes, so it is the one
#: this rule MUST cover. tilefoundry_gate.py still holds two direct calls
#: (container_exists: docker inspect without a timeout, tool_version: timeout+wait on a
#: child that may be in D state); that is recorded as C9 and is fixed in that batch, not
#: silently allowlisted here -- adding it to ALLOWLIST would hide exactly the hazard
#: this test exists for.
DEVICE_MODULES = ("verify_environment.py",)

#: Sites still allowed to call subprocess.run directly. Empty on purpose: a new entry
#: has to be justified in review, which is the point.
ALLOWLIST: dict = {}


def bare_run_calls(path: Path) -> list[int]:
    """Line numbers of real `subprocess.run(...)` CALLS in *path*.

    Read from the AST, not from the text: a docstring or a comment may NAME the
    pattern (this very module explains it), and a text search then reports prose as
    a violation. A checker that fails on its own documentation gets weakened, so
    the check looks at calls only.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "run":
            continue
        owner = func.value
        name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
        if name == "subprocess":
            sites.append(node.lineno)
    return sorted(sites)


def test_no_bare_subprocess_run_in_device_modules() -> None:
    offenders = []
    for name in DEVICE_MODULES:
        path = SCRIPTS / name
        if not path.is_file():
            continue
        if name in ALLOWLIST:
            continue
        for number in bare_run_calls(path):
            offenders.append("%s:%d" % (name, number))
    assert not offenders, (
        "device modules must call bounded_run(), not subprocess.run(timeout=...), "
        "because that timeout does not bound a D-state child: " + ", ".join(offenders))


def test_verify_environment_honours_the_test_lock_override() -> None:
    """Tests must be able to run without taking the canonical team lock.

    /home/.tf_device0.lock coordinates the whole team. A test suite that takes it
    can block a real measurement job (or be blocked by one), so the module must
    read TF_DEVICE_LOCK_PATH and record whether the path it used was canonical.
    """
    text = (SCRIPTS / "verify_environment.py").read_text()
    assert "TF_DEVICE_LOCK_PATH" in text, (
        "verify_environment.py has no test-only lock override, so running its "
        "negatives would take the canonical team lock")
    assert "lock_canonical" in text, (
        "the run must RECORD whether the lock was the canonical one: a private "
        "lock proves the mechanism, not team serialisation")


def test_bounded_run_is_the_one_helper() -> None:
    text = (SCRIPTS / "verify_environment.py").read_text()
    assert "def bounded_run(" in text
    assert "killpg" in text, "the child must be killed as a PROCESS GROUP"
    assert "unreaped" in text, "an unkillable child must be REPORTED, not waited on"
