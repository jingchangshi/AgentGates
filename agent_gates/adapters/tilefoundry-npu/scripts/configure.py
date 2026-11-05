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
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_context as E  # noqa: E402


class ConfigureError(RuntimeError):
    """A NAMED failure. The caller prints the code and exits non-zero.

    The previous revision returned 0 from the converge path no matter what the
    actions said, and represented a failed metadata read as an empty package set.
    A named exception is what makes "failed" impossible to mistake for "done".
    """

    def __init__(self, code: str, message: str, exit_code: int = 1):
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.exit_code = exit_code


def _run(command, *, env=None, timeout=3600, cwd=None):
    """Run a command, returning (rc, stdout, stderr) -- never raising on failure.

    Callers report raw output and decide what a non-zero means. There is no
    generic "check" shortcut here: a shared fallback turned every distinct fault
    into the same SystemExit, which is how a specific refusal gets lost.
    """
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
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
    """{normalised name: version} as the SHARED interpreter sees them.

    A READ FAILURE IS NOT AN EMPTY SET. Returning {} here made the plan see every
    pin as missing, so a broken interpreter was "repaired" by reinstalling the
    whole lock into it -- the one action that cannot work and hides the fault.
    Reading is therefore fallible by name, and the caller must refuse to install
    on it (INSTALLED_METADATA_UNREADABLE / INTERPRETER_MISSING).
    """
    if not _interpreter_ready():
        raise ConfigureError(
            "INTERPRETER_MISSING",
            "%s does not exist or is not executable" % E.interpreter(), 2)
    code = (
        "import json,sys;"
        "import importlib.metadata as m;"
        "d={};"
        "[d.__setitem__(dist.metadata['Name'], dist.version) for dist in m.distributions()];"
        "print(json.dumps(d))"
    )
    rc, out, err = _run([str(E.interpreter()), "-c", code], timeout=300)
    if rc != 0:
        raise ConfigureError(
            "INSTALLED_METADATA_UNREADABLE",
            "the interpreter could not enumerate installed packages (rc=%s): %s"
            % (rc, ((err or "") + (out or "")).strip()[-300:]), 2)
    for line in reversed((out or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
    raise ConfigureError(
        "INSTALLED_METADATA_UNREADABLE",
        "the interpreter printed no JSON package metadata: %r"
        % ((out or "").strip()[-200:],), 2)


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

#: The domestic mirror. It is reachable from this host WITHOUT the proxy, so using
#: the proxy against it was simply wrong: the proxy exists for slow/unreachable
#: OVERSEAS addresses, and adding it to a domestic request only adds a hop.
MIRROR_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
MIRROR_HOST = "mirrors.aliyun.com"

#: Packages the domestic mirror does NOT publish. Only these may use the proxy,
#: and only after the mirror was actually tried without it. Measured case: torch's
#: CPU build is served by download.pytorch.org, not by the PyPI mirror, so the
#: mirror cannot satisfy the pin no matter how long it is waited on.
OVERSEAS_INDEX = {
    "torch": "https://download-r2.pytorch.org/whl/cpu",
}

#: What the network half actually did. Recorded in facts so a run can say
#: "proxy_used=false" and mean it, instead of leaving the reader to guess.
NETWORK = {"proxy_used": False, "packages": [], "reason": "offline", "indexes": []}


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


def _network_facts() -> dict:
    """A copy of what the network half did, for the facts report."""
    return json.loads(json.dumps(NETWORK))


def _warm_cache(names, steps, *, index, host, proxy, env, actions):
    """Put the fetched wheels in the local cache, so the NEXT run is offline.

    A failed warm is RECORDED rather than dropped: if the wheel is not in
    envs/wheels, the next offline run will name it missing, and the reason has to
    already exist in the evidence rather than be reconstructed later.
    """
    for name in names:
        want = next(step["want"] for step in steps if step["name"] == name)
        argv = [str(E.interpreter()), "-m", "pip", "download", "--no-deps",
                "-i", index, "--trusted-host", host,
                "--timeout", str(PIP_TIMEOUT), "--retries", str(PIP_RETRIES),
                "-d", str(E.WHEEL_CACHE), "%s==%s" % (name, want)]
        command = argv
        if proxy is not None:
            command = ["bash", "-lc", ". %s >/dev/null 2>&1; exec %s"
                       % (proxy, " ".join(argv))]
        try:
            rc, out, err = _run(command, env=env, timeout=INSTALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            actions.append({"action": "cache_warm", "code": "CONFIGURE_NETWORK_TIMEOUT",
                            "wheel": name, "index": index,
                            "timeout_s": INSTALL_TIMEOUT})
            continue
        actions.append({"action": "cache_warm", "wheel": name, "rc": rc,
                        "index": index,
                        "proxy": str(proxy) if proxy is not None else None,
                        "tail": (out + err)[-300:]})


def _install(steps: list[dict]):
    """Converge on the plan: OFFLINE, then the domestic mirror, then -- only for a
    wheel the mirror genuinely does not publish -- the upstream index via proxy.

    The order is a contract, not a preference:
      1. offline is the reproducibility claim (the local cache covers the pins);
      2. the domestic mirror is reachable DIRECTLY. Routing it through the proxy
         was a real defect: the proxy exists for slow/unreachable OVERSEAS
         addresses, and adding it to a domestic request only adds a hop;
      3. the proxy is used only for packages whose wheel lives upstream
         (OVERSEAS_INDEX), and facts record proxy_used=true plus those names, so
         "the internet was needed" is never a vague statement.

    Returns (actions, failure); failure is a NAMED dict or None.
    """
    if not steps:
        return [], None
    recipes = ["%s==%s" % (step["name"], step["want"]) for step in steps]
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    actions = []
    NETWORK.update({"proxy_used": False, "packages": [], "reason": "offline",
                    "indexes": []})

    # ---- 1. offline, bounded
    offline = _pip_argv()
    if E.WHEEL_CACHE.is_dir():
        offline += ["--no-index", "--find-links", str(E.WHEEL_CACHE)]
    offline += recipes
    started = time.time()
    try:
        rc, out, err = _run(offline, env=env, timeout=INSTALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        actions.append({"action": "pip", "mode": "offline",
                        "code": "CONFIGURE_NETWORK_TIMEOUT", "rc": None,
                        "timeout_s": INSTALL_TIMEOUT, "argv": offline})
        return actions, {"code": "CONFIGURE_NETWORK_TIMEOUT",
                         "detail": "the offline install exceeded %ss" % INSTALL_TIMEOUT}
    actions.append({"action": "pip", "mode": "offline", "rc": rc,
                    "seconds": round(time.time() - started, 2), "argv": offline,
                    "tail": (out + err)[-600:]})
    if rc == 0:
        NETWORK["reason"] = "offline"
        return actions, None

    missing = _missing_wheels(err, steps)
    actions.append({"action": "diagnose", "code": "WHEEL_MISSING", "missing": missing,
                    "cache": str(E.WHEEL_CACHE)})
    if not missing:
        return actions, {
            "code": "CONFIGURE_INSTALL_FAILED",
            "detail": "the offline install failed without naming a missing wheel, so "
                      "there is nothing specific to fetch (rc=%s)" % rc}
    if os.environ.get("TFNU_OFFLINE_ONLY", "").strip() == "1":
        return actions, {"code": "WHEEL_MISSING",
                         "detail": "offline only, and the local wheel cache lacks: %r"
                                   % (missing,)}

    missing_steps = [step for step in steps if step["name"] in missing]

    # ---- 2. domestic mirror, explicitly WITHOUT the proxy
    mirror = _pip_argv() + ["-i", MIRROR_INDEX, "--trusted-host", MIRROR_HOST]
    mirror += ["%s==%s" % (step["name"], step["want"]) for step in missing_steps]
    started = time.time()
    try:
        rc, out, err = _run(mirror, env=env, timeout=INSTALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        actions.append({"action": "pip", "mode": "mirror", "proxy": False,
                        "code": "CONFIGURE_NETWORK_TIMEOUT", "rc": None,
                        "timeout_s": INSTALL_TIMEOUT, "argv": mirror})
        return actions, {"code": "CONFIGURE_NETWORK_TIMEOUT",
                         "detail": "the domestic mirror install exceeded %ss"
                                   % INSTALL_TIMEOUT}
    actions.append({"action": "pip", "mode": "mirror", "proxy": False,
                    "index": MIRROR_INDEX, "rc": rc,
                    "seconds": round(time.time() - started, 2), "argv": mirror,
                    "tail": (out + err)[-600:]})
    NETWORK.update({"proxy_used": False, "packages": [],
                    "reason": "domestic mirror (no proxy)", "indexes": [MIRROR_INDEX]})
    if rc == 0:
        _warm_cache(missing, steps, index=MIRROR_INDEX, host=MIRROR_HOST,
                    proxy=None, env=env, actions=actions)
        return actions, None

    still = _missing_wheels(err, missing_steps)
    actions.append({"action": "diagnose", "code": "MIRROR_INCOMPLETE",
                    "missing": still, "index": MIRROR_INDEX})
    if not still:
        return actions, {
            "code": "CONFIGURE_MIRROR_FAILED",
            "detail": "the domestic mirror attempt failed without naming a missing "
                      "wheel (rc=%s)" % rc}
    overseas = {name: OVERSEAS_INDEX[name] for name in still
                if name in OVERSEAS_INDEX}
    unfetchable = [name for name in still if name not in OVERSEAS_INDEX]
    if unfetchable or not overseas:
        return actions, {
            "code": "WHEEL_MISSING",
            "detail": "not in the local wheel cache and not published on the "
                      "domestic mirror: %r" % (unfetchable or still,)}

    # ---- 3. upstream, proxy ONLY for the named overseas packages
    proxy = Path(PROXY_SCRIPT)
    if not proxy.is_file():
        return actions, {
            "code": "PROXY_MISSING_FOR_OVERSEAS",
            "detail": "%r is only served by the upstream index, and %s does not "
                      "exist" % (sorted(overseas), PROXY_SCRIPT)}
    for name in sorted(overseas):
        index = overseas[name]
        host = urlparse(index).hostname or ""
        want = next(step["want"] for step in steps if step["name"] == name)
        argv = _pip_argv() + ["-i", index, "--trusted-host", host,
                              "%s==%s" % (name, want)]
        command = ["bash", "-lc", ". %s >/dev/null 2>&1; exec %s"
                   % (proxy, " ".join(argv))]
        started = time.time()
        try:
            rc, out, err = _run(command, env=env, timeout=INSTALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            actions.append({"action": "pip", "mode": "overseas", "proxy": True,
                            "code": "CONFIGURE_NETWORK_TIMEOUT", "rc": None,
                            "timeout_s": INSTALL_TIMEOUT, "packages": [name],
                            "index": index, "argv": argv})
            return actions, {"code": "CONFIGURE_NETWORK_TIMEOUT",
                             "detail": "the upstream install of %r exceeded %ss"
                                       % (name, INSTALL_TIMEOUT)}
        actions.append({"action": "pip", "mode": "overseas", "proxy": True,
                        "index": index, "packages": [name], "rc": rc,
                        "seconds": round(time.time() - started, 2), "argv": argv,
                        "tail": (out + err)[-600:]})
        if rc != 0:
            return actions, {
                "code": "CONFIGURE_OVERSEAS_INSTALL_FAILED",
                "detail": "%s is not on the domestic mirror and its upstream "
                          "install failed (index=%s, rc=%s)" % (name, index, rc)}
        _warm_cache([name], steps, index=index, host=host, proxy=proxy,
                    env=env, actions=actions)

    NETWORK.update({"proxy_used": True, "packages": sorted(overseas),
                    "reason": "not published on the domestic mirror; the upstream "
                              "index serves it", "indexes": sorted(set(overseas.values()))})
    return actions, None


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
    try:
        installed = _installed()
        installed_error = None
    except ConfigureError as exc:
        # FACTS MUST STILL BE PRODUCED: the whole point of --facts on a broken
        # prefix is to show WHY it is broken. What must NOT happen is treating the
        # empty map as "every pin is missing".
        installed = {}
        installed_error = {"code": exc.code, "detail": str(exc)}
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
        # An UNREADABLE package set is reported as unreadable, never as "all pins
        # missing": the latter is indistinguishable from a fresh prefix and would
        # justify reinstalling a whole lock into a broken interpreter.
        "installed_readable": installed_error is None,
        "installed_error": installed_error,
        "pin_problems": (E.pin_problems(installed) if installed_error is None
                         else {"code": installed_error["code"],
                               "missing": None,
                               "detail": "not evaluated: package metadata unreadable"}),
        "installed_count": len(installed) if installed_error is None else None,
        # Which network tier ran, and whether the proxy was needed. Recorded, not
        # inferred: "proxy_used" must be a fact about this run.
        "network": _network_facts(),
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


def _fail(actions: list, code: str, detail: str, exit_code: int = 1) -> int:
    """Print a NAMED failure and return a NON-ZERO code.

    The converge path used to end in a hard-coded `return 0`: every failure was
    printed into the action list and then the program reported success. A caller
    (the verify phase, a run recipe) could not tell a converged prefix from a
    broken one, so "configure exited 0" meant nothing.
    """
    report = {"actions": actions, "verdict": "FAIL", "code": code, "detail": detail}
    print(json.dumps(report, indent=1, sort_keys=True))
    print("%s: %s" % (code, detail), file=sys.stderr)
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", action="store_true", help="inspect only, write nothing")
    parser.add_argument("--findings", action="store_true", help="exit non-zero if unusable")
    args = parser.parse_args(argv)

    if args.facts:
        facts = gather_facts()
        print(json.dumps(facts, indent=1, sort_keys=True))
        # The facts WERE produced, including an unreadable-metadata fact. The
        # non-zero refusal belongs to the paths that could otherwise repair or
        # approve the prefix (converge, --findings).
        return 0

    actions = []
    if args.findings:
        # A CHECK MUST NOT MUTATE. Measured defect this prevents: --findings was
        # run to confirm the prefix, one pin disagreed, and the check silently
        # reinstalled that pin -- reverting a diagnosed downgrade and breaking the
        # very import it had been asked to confirm. Inspect, report, exit.
        try:
            installed = _installed()
        except ConfigureError as exc:
            actions.append({"action": "refused", "code": exc.code,
                            "detail": str(exc)})
            return _fail(actions, exc.code, str(exc), exc.exit_code)
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

    # ---- converge: every failure stops the sequence, and none of them exits 0.
    try:
        if not _interpreter_ready():
            created = _create_prefix()
            actions.append(created)
            if created.get("rc") != 0:
                return _fail(
                    actions, "CONFIGURE_PREFIX_CREATE_FAILED",
                    "creating %s failed (rc=%r): %s"
                    % (E.SHARED_PREFIX, created.get("rc"), created.get("tail")))

        installed = _installed()
        steps = _plan(installed)
        if steps:
            actions.append({"action": "plan", "steps": steps})
            install_actions, failure = _install(steps)
            actions.extend(install_actions)
            if failure is not None:
                # Nothing further runs: not the re-read, not facts that look like a
                # verdict, and certainly not a second install attempt.
                return _fail(actions, failure["code"], failure["detail"])
            # A successful pip is not a satisfied pin. Re-read the interpreter and
            # re-plan: an install that reported success while leaving the pins wrong
            # must fail, or "converged" would be a claim about pip's exit code.
            installed = _installed()
            remaining = _plan(installed)
            if remaining:
                return _fail(
                    actions, "CONFIGURE_PINS_UNSATISFIED",
                    "after the install these pins are still wrong: %r" % (remaining,))
        else:
            # Idempotence is an assertion, not a hope: with an empty plan nothing is
            # written, so a second run genuinely changes nothing.
            actions.append({"action": "noop", "reason": "every pin already satisfied"})
    except ConfigureError as exc:
        return _fail(actions, exc.code, str(exc), exc.exit_code)

    facts = gather_facts()
    report = {"actions": actions, "facts": facts}
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
