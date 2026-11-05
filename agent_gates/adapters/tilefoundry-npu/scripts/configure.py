"""Provision the shared interpreter the TileFoundry adapter runs against.

One command, idempotent by construction: it INSPECTS first and only acts on what
is actually missing or wrong. A second run must write nothing, which is the only
way "already provisioned" is a claim anyone can check.

What it does NOT do: rebuild an image, install a driver, change CANN, or touch the
system interpreter. Everything lands under the shared workspace prefix, which the
host and the container both see at the same path.

Usage:
    python configure.py --facts         # inspect only, print facts, change nothing
    python configure.py                 # converge (install only what is missing)
    python configure.py --findings      # inspect ONLY, exit non-zero if unusable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_context as E  # noqa: E402


def _run(command, *, env=None, timeout=3600, cwd=None, check=False):
    """Run a command, returning (rc, stdout, stderr) -- never raising on failure.

    Callers report raw output. A silent non-zero here would be the exact defect
    this module exists to make visible.
    """
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
    )
    if check and done.returncode != 0:
        raise SystemExit(
            "command failed (%s): %s\n%s\n%s"
            % (done.returncode, " ".join(map(str, command)), done.stdout, done.stderr)
        )
    return done.returncode, done.stdout, done.stderr


def _interpreter_ready() -> bool:
    target = E.interpreter()
    return target.exists() and os.access(str(target), os.X_OK)


def _create_prefix() -> dict:
    """Create the venv. Only ever called when it does not exist."""
    base = os.environ.get("TFNU_BASE_PYTHON") or sys.executable
    E.SHARED_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    E.WHEEL_CACHE.mkdir(parents=True, exist_ok=True)
    rc, out, err = _run([base, "-m", "venv", str(E.SHARED_PREFIX)])
    return {"action": "create_prefix", "base": base, "rc": rc, "tail": (out + err)[-400:]}


def _installed() -> dict:
    """{normalised name: version} as the SHARED interpreter sees them."""
    if not _interpreter_ready():
        return {}
    code = (
        "import json,sys;"
        "import importlib.metadata as m;"
        "d={};"
        "[d.__setitem__(dist.metadata['Name'], dist.version) for dist in m.distributions()];"
        "print(json.dumps(d))"
    )
    rc, out, err = _run([str(E.interpreter()), "-c", code], timeout=300)
    if rc != 0:
        return {}
    for line in reversed((out or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def _plan(installed: dict) -> list[dict]:
    """What is missing or wrong. Empty list means nothing to do."""
    steps = []
    for name, want in sorted(E.ALL_PINS.items()):
        have = installed.get(name)
        if have is None:
            steps.append({"kind": "install", "name": name, "want": want, "have": None})
        elif have != want:
            steps.append({"kind": "reinstall", "name": name, "want": want, "have": have})
    return steps


def _wheel_for(name: str, version: str | None = None) -> Path | None:
    """The cached wheel matching the pin, if we already have one.

    The VERSION must match too. Matching only the name was a real defect: with a
    pin move (z3-solver 5.3.3.0 -> 4.15.4.0) the old wheel still matched the
    prefix, so the plan went to pip with --no-index against a cache that could
    not satisfy it. Offline must mean "the pinned artefact is here", not "some
    artefact with that name is here".
    """
    if not E.WHEEL_CACHE.is_dir():
        return None
    stem = name.lower().replace("-", "_").replace(".", "_")
    want = (version or "").lower().replace(".", "_").replace("-", "_")
    for candidate in sorted(E.WHEEL_CACHE.glob("*.whl")):
        # Normalise DOTS too: the version in a filename keeps them ("pyyaml_6.0.3"),
        # while the pin is compared as "6_0_3". Without this every pin looks absent
        # offline, which would send a converge to the network for no reason.
        low = candidate.name.lower().replace("-", "_").replace(".", "_")
        if not (low.startswith(stem + "-") or low.startswith(stem + "_")):
            continue
        if not want:
            return candidate
        if low.startswith("%s-%s" % (stem.replace("_", "-"), want)):
            return candidate
        if low.startswith("%s_%s" % (stem, want)):
            return candidate
    return None


#: Bounds on the installer. A pip that waits on a network nobody can reach is
#: indistinguishable from a hung program; these turn it into a NAMED result.
PIP_TIMEOUT = os.environ.get("TFNU_PIP_TIMEOUT", "60")
PIP_RETRIES = os.environ.get("TFNU_PIP_RETRIES", "2")
INSTALL_TIMEOUT = int(os.environ.get("TFNU_INSTALL_TIMEOUT", "900"))
PROXY_SCRIPT = os.environ.get("TFNU_PROXY_SH", os.path.expanduser("~/set_proxy.sh"))
MIRROR_INDEX = "https://mirrors.aliyun.com/pypi/simple/"


def _pip_argv() -> list:
    return [str(E.interpreter()), "-m", "pip", "install", "--no-input",
            "--no-warn-script-location", "--timeout", str(PIP_TIMEOUT),
            "--retries", str(PIP_RETRIES)]


def _missing_wheels(stderr: str, steps: list[dict]) -> list[str]:
    """Which pinned packages the offline attempt could not satisfy.

    Named, not "offline install failed": the point is to say WHICH wheel is not in
    the local cache, because that is the only fact that makes networking necessary.
    """
    text = stderr or ""
    missing = []
    for step in steps:
        name = step["name"].lower().replace("_", "-")
        if ("no matching distribution found for %s==" % step["name"]).lower() in text.lower():
            missing.append(step["name"])
        elif name in text.lower() and "could not find a version" in text.lower():
            if step["name"] not in missing:
                missing.append(step["name"])
    return missing


def _install(steps: list[dict]) -> list[dict]:
    """Converge on the plan: OFFLINE FIRST, network only when a wheel is missing.

    Offline is not a preference, it is the reproducibility claim: 35 wheels in the
    local cache cover every pin, so a converge must finish in milliseconds. If it
    cannot, the answer is "WHEEL_MISSING: <name>" -- not a silent wait on a mirror.
    """
    if not steps:
        return []
    recipes = ["%s==%s" % (step["name"], step["want"]) for step in steps]
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    actions = []

    # ---- 1. offline, bounded
    offline = _pip_argv()
    if E.WHEEL_CACHE.is_dir():
        offline += ["--no-index", "--find-links", str(E.WHEEL_CACHE)]
    offline += recipes
    started = time.time()
    try:
        rc, out, err = _run(offline, env=env, timeout=INSTALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return [{"action": "pip", "mode": "offline", "code": "CONFIGURE_NETWORK_TIMEOUT",
                 "rc": None, "timeout_s": INSTALL_TIMEOUT, "argv": offline}]
    actions.append({"action": "pip", "mode": "offline", "rc": rc,
                    "seconds": round(time.time() - started, 2), "argv": offline,
                    "tail": (out + err)[-600:]})
    if rc == 0:
        return actions

    missing = _missing_wheels(err, steps)
    actions.append({"action": "diagnose", "code": "WHEEL_MISSING", "missing": missing,
                    "cache": str(E.WHEEL_CACHE)})
    if not missing or os.environ.get("TFNU_OFFLINE_ONLY", "").strip() == "1":
        return actions

    # ---- 2. network, ONLY for the named missing wheels, and bounded
    online = _pip_argv() + ["-i", MIRROR_INDEX, "--trusted-host", "mirrors.aliyun.com"]
    online += ["%s==%s" % (step["name"], step["want"])
               for step in steps if step["name"] in missing]
    proxy = Path(PROXY_SCRIPT)
    command = online
    if proxy.is_file():
        command = ["bash", "-lc", ". %s >/dev/null 2>&1; exec %s"
                   % (proxy, " ".join(online))]
    started = time.time()
    try:
        rc, out, err = _run(command, env=env, timeout=INSTALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        actions.append({"action": "pip", "mode": "online",
                        "code": "CONFIGURE_NETWORK_TIMEOUT", "rc": None,
                        "timeout_s": INSTALL_TIMEOUT, "argv": online})
        return actions
    actions.append({"action": "pip", "mode": "online", "rc": rc,
                    "seconds": round(time.time() - started, 2), "argv": online,
                    "proxy": str(proxy) if proxy.is_file() else None,
                    "tail": (out + err)[-600:]})
    if rc == 0:
        # Persist what was fetched, so the NEXT run is offline again. Without this the
        # network becomes a permanent dependency and this whole path is unrepeatable.
        for name in missing:
            want = next(step["want"] for step in steps if step["name"] == name)
            _run([str(E.interpreter()), "-m", "pip", "download", "--no-deps",
                  "-i", MIRROR_INDEX, "--trusted-host", "mirrors.aliyun.com",
                  "--timeout", str(PIP_TIMEOUT), "--retries", str(PIP_RETRIES),
                  "-d", str(E.WHEEL_CACHE), "%s==%s" % (name, want)],
                 env=env, timeout=INSTALL_TIMEOUT)
        actions.append({"action": "cache_warmed", "wheels": missing,
                        "cache": str(E.WHEEL_CACHE)})
    return actions


def _tilelang_state() -> dict:
    """Source revision and dirtiness -- a wheel from an unknown tree is not pinned."""
    root = E.TILELANG_ROOT
    state = {"root": str(root), "commit": None, "dirty": None}
    if not (root / ".git").exists():
        state["error"] = "not a git checkout"
        return state
    rc, out, _ = _run(["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"], timeout=60)
    if rc == 0:
        state["commit"] = out.strip()
    rc, out, _ = _run(["git", "-C", str(root), "status", "--porcelain"], timeout=60)
    if rc == 0:
        state["dirty"] = bool(out.strip())
        state["dirty_paths"] = out.strip().splitlines()[:10]
    return state


def _wheel_facts() -> dict:
    """The installed tilelang wheel: name and sha256, so a rebuild is verifiable."""
    facts = {"wheel_cache": str(E.WHEEL_CACHE), "wheels": []}
    if E.WHEEL_CACHE.is_dir():
        for path in sorted(E.WHEEL_CACHE.glob("tilelang*.whl")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            facts["wheels"].append(
                {"name": path.name, "sha256": digest, "bytes": path.stat().st_size}
            )
    return facts


def gather_facts() -> dict:
    """Everything a reader needs to decide whether this prefix is trustworthy."""
    installed = _installed()
    facts = {
        "workspace": str(E.WORKSPACE),
        "prefix": str(E.SHARED_PREFIX),
        "prefix_exists": E.SHARED_PREFIX.is_dir(),
        # Which environment this run is about. v2 (main) is the default; the v1
        # rollback prefix is legitimate but must be VISIBLE, so it is a fact, not a
        # local convention.
        "env": {
            "prefix": str(E.SHARED_PREFIX),
            "rollback_prefix": E.prefix_is_rollback(),
            "rollback_prefix_path": str(E.ROLLBACK_PREFIX),
            "expected_tilelang_wheel_version": E.TILELANG_WHEEL_VERSION,
            "tilelang_source_commit": E.TILELANG_COMMIT,
            # RECORDED, never used to decide pass: a pass obtained with the source
            # tree prepended is not a pass about the deployed wheel.
            "source_tree_prepended": (os.environ.get(E.SOURCE_TREE_PREPENDED_ENV) == "1"
                                     or E.source_tree_allowed()),
            "allow_source_tree_env": E.ALLOW_SOURCE_TREE_ENV,
        },
        "interpreter": str(E.interpreter()),
        "interpreter_exists": _interpreter_ready(),
        "python_version": None,
        "pythonpath_entries": E.python_path_entries(),
        "pins": E.ALL_PINS,
        "pin_problems": E.pin_problems(installed),
        "installed_count": len(installed),
        "tilelang_source": _tilelang_state(),
        "tilelang_wheel": _wheel_facts(),
        "cann_set_env": E.CANN_SET_ENV,
        # CANN is a PRECONDITION of the import layers, not a neighbouring
        # convenience: torch_npu dlopens libhccl.so at import time and torch's
        # device-backend autoload turns that into "Failed to load the backend
        # extension: torch_npu", which reads as "tilelang is broken".
        "cann": E.cann_state(),
        # z3's version is dictated by the wheel's libtvm_compiler.so SONAME; the
        # intersection with tilelang's own declared range is asserted, not assumed.
        "z3_requirement": E.z3_requirement(installed),
        # The CCE compiler is external to tilelang (see env_context.TOOLCHAIN).
        # Recorded here because a missing one only fails LATE, at compile-to-binary.
        "toolchain": E.probe_toolchain(),
    }
    if facts["interpreter_exists"]:
        rc, out, _ = _run([str(E.interpreter()), "-V"], timeout=60)
        if rc == 0:
            facts["python_version"] = out.strip()
    facts["probe"] = E.run_probe() if facts["interpreter_exists"] else {"ok": False}
    # Negative control: with CANN stripped, the import must fail NAMING the CANN
    # gap. If it fails with an anonymous .so error instead, the positive probe's
    # green is worth less -- that is the point of running both.
    facts["cann_negative_probe"] = (
        E.negative_probe() if facts["interpreter_exists"] else {"ok": False}
    )
    return facts


def _findings(facts: dict) -> int:
    """Exit non-zero with a NAMED reason, in dependency order.

    Order is the contract: CANN first (it gates every import), then the import
    layers, then the z3 intersection (it gates the shared object), then pins.
    Reporting a later failure first would name a symptom of an earlier cause.
    """
    if not facts.get("interpreter_exists"):
        print(
            "NO INTERPRETER: %s does not exist; every pin below is unverifiable"
            % facts.get("interpreter"),
            file=sys.stderr,
        )
        return 1
    env_facts = facts.get("env") or {}
    if env_facts.get("source_tree_prepended"):
        print("SOURCE TREE ENABLED: " + str(env_facts.get("allow_source_tree_env")) + "=1 -- the import path prepends the tilelang checkout, so this environment must not be reported as verified. Unset it and re-run.", file=sys.stderr)
        return 1
    cann = facts.get("cann") or {}
    if not cann.get("sourced"):
        print("CANN NOT SOURCED:", cann.get("code"), "-", cann.get("consequence"), file=sys.stderr)
        print("  source %s first" % facts.get("cann_set_env"), file=sys.stderr)
        return 1
    probe = facts.get("probe") or {}
    if not probe.get("ok"):
        code = probe.get("code") or "PROBE_FAILED"
        print("UNUSABLE:", code, "-", probe.get("error"), file=sys.stderr)
        return 1
    negative = facts.get("cann_negative_probe") or {}
    if not negative.get("ok"):
        print(
            "NEGATIVE CONTROL FAILED: without CANN the import must report %s, got %r"
            % (negative.get("expected_code"), negative.get("observed_code")),
            file=sys.stderr,
        )
        return 1
    z3 = facts.get("z3_requirement") or {}
    if not z3.get("ok"):
        print(
            "Z3 OFF: %s installed=%s pin=%s requirement=%r source=%s"
            % (z3.get("code"), z3.get("installed"), z3.get("pin"),
               z3.get("requirement"), z3.get("requirement_source")),
            file=sys.stderr,
        )
        return 1
    if facts.get("pin_problems"):
        print("PINS OFF:", json.dumps(facts["pin_problems"]), file=sys.stderr)
        return 1
    if not (facts.get("toolchain") or {}).get("ok"):
        print("TOOLCHAIN UNRESOLVED:", (facts.get("toolchain") or {}).get("code"), file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", action="store_true", help="inspect only, write nothing")
    parser.add_argument("--findings", action="store_true", help="exit non-zero if unusable")
    args = parser.parse_args(argv)

    if args.facts:
        print(json.dumps(gather_facts(), indent=1, sort_keys=True))
        return 0

    actions = []
    if args.findings:
        # A CHECK MUST NOT MUTATE. Measured defect this prevents: --findings was
        # run to confirm the prefix, one pin disagreed, and the check silently
        # reinstalled that pin -- reverting a diagnosed downgrade and breaking the
        # very import it had been asked to confirm. Inspect, report, exit.
        installed = _installed()
        pending = _plan(installed)
        if pending:
            # Report the drift; do not "fix" it behind the caller's back.
            actions.append({"action": "inspect_only", "would_install": pending,
                            "reason": "--findings never installs"})
        else:
            actions.append({"action": "noop",
                            "reason": "every pin already satisfied (--findings does not install)"})
        facts = gather_facts()
        report = {"actions": actions, "facts": facts}
        print(json.dumps(report, indent=1, sort_keys=True))
        return _findings(facts)

    if not _interpreter_ready():
        actions.append(_create_prefix())

    installed = _installed()
    steps = _plan(installed)
    if steps:
        actions.append({"action": "plan", "steps": steps})
        actions.extend(_install(steps))
    else:
        # Idempotence is an assertion, not a hope: with an empty plan nothing is
        # written, so a second run genuinely changes nothing.
        actions.append({"action": "noop", "reason": "every pin already satisfied"})

    facts = gather_facts()
    report = {"actions": actions, "facts": facts}
    print(json.dumps(report, indent=1, sort_keys=True))

    if args.findings:
        return _findings(facts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
