"""C11 negatives for configure.py: a failure must be NAMED, non-zero, and final.

Three real defects are pinned here, each measured in review:

  * `_installed()` returned {} when the metadata command failed. The plan then saw
    every pin as missing, so a broken interpreter was "repaired" by reinstalling the
    whole lock into it -- the one action that cannot work, and the one that hides the
    fault. A read failure is now INSTALLED_METADATA_UNREADABLE and NO install runs.
  * the converge path ended in a hard-coded `return 0`, so pip's failure was printed
    into the action list and the program still reported success.
  * a failed pip could be followed by another install attempt. A failure STOPS the
    sequence: nothing is retried, and the failed steps are not re-planned as if the
    environment had moved.

They also pin the network tiers: offline, then the domestic mirror WITHOUT the
proxy, then the proxy only for a wheel the mirror genuinely does not publish
(the proxy exists for unreachable OVERSEAS addresses, not for the mirror).

No test touches the real interpreter or the network: `_run` is replaced.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"
sys.path.insert(0, str(SCRIPTS))

import configure as C  # noqa: E402


TORCH = "2.12.0+cpu"


@pytest.fixture
def one_pin(monkeypatch, tmp_path):
    """Exactly one pin, and a wheel cache that exists but does not satisfy it."""
    monkeypatch.setattr(C.E, "ALL_PINS", {"torch": TORCH})
    monkeypatch.setattr(C.E, "WHEEL_CACHE", tmp_path / "wheels")
    (tmp_path / "wheels").mkdir()
    monkeypatch.setattr(C, "_interpreter_ready", lambda: True)
    monkeypatch.delenv("TFNU_OFFLINE_ONLY", raising=False)
    monkeypatch.setattr(C, "gather_facts", lambda: {"patched": True})
    return tmp_path


def _run_main():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = C.main([])
    return rc, out.getvalue() + err.getvalue()


def _report(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise AssertionError("no JSON report was printed:\n" + text)
    return json.JSONDecoder().raw_decode(text[start:])[0]


def test_a_failed_metadata_read_is_not_an_empty_environment(monkeypatch):
    """A broken read must be NAMED and must trigger NO install."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return 1, "", "metadata exploded"

    monkeypatch.setattr(C, "_interpreter_ready", lambda: True)
    monkeypatch.setattr(C, "_run", fake_run)

    rc, text = _run_main()
    report = _report(text)

    assert rc == 2, "a failed metadata read must exit non-zero (got %r)" % rc
    assert report["code"] == "INSTALLED_METADATA_UNREADABLE", report
    assert not [call for call in calls if "pip" in call], (
        "a failed metadata read must NOT be read as 'every pin is missing': "
        "no pip install may run. Calls: %r" % (calls,))


def test_pip_failure_is_named_and_does_not_reinstall(one_pin, monkeypatch):
    """One offline pip failure -> one named refusal, exactly one pip attempt."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return (1, "", "ERROR: No matching distribution found for torch==%s" % TORCH)

    monkeypatch.setattr(C, "_installed", lambda: {})
    monkeypatch.setattr(C, "_run", fake_run)
    # Offline-only makes the FIRST failure final, which is exactly the property
    # under test: no second attempt, no reinstall, no fall-through to another tier.
    monkeypatch.setenv("TFNU_OFFLINE_ONLY", "1")
    rc, text = _run_main()
    report = _report(text)

    assert rc != 0, "a failed install must not exit 0"
    assert report["code"] == "WHEEL_MISSING", report
    pip_calls = [call for call in calls if "pip" in call]
    assert len(pip_calls) == 1, (
        "a failed pip must not be followed by another install attempt: %r" % (pip_calls,))


def test_the_mirror_is_used_without_the_proxy(one_pin, monkeypatch):
    """Tier 2: the domestic mirror succeeds, and the proxy is NOT involved."""
    monkeypatch.setattr(C, "PROXY_SCRIPT", str(one_pin / "no-proxy-here.sh"))
    calls = []
    installed_after = {"torch": TORCH}
    reads = {"n": 0}

    def fake_installed():
        reads["n"] += 1
        return {} if reads["n"] == 1 else dict(installed_after)

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        joined = " ".join(argv)
        if "mirrors.aliyun.com" in joined:
            return 0, "", ""
        if "download" in argv:
            return 0, "", ""
        return (1, "", "ERROR: No matching distribution found for torch==%s" % TORCH)

    monkeypatch.setattr(C, "_installed", fake_installed)
    monkeypatch.setattr(C, "_run", fake_run)

    rc, text = _run_main()
    report = _report(text)

    assert rc == 0, report
    assert report["facts"]["patched"] is True
    mirror_calls = [call for call in calls if "mirrors.aliyun.com" in " ".join(call)]
    assert mirror_calls, "the domestic mirror tier was never tried: %r" % (calls,)
    assert all(call[0] != "bash" for call in calls), (
        "the mirror must be reached directly; a shell was used: %r" % (calls,))
    assert C.NETWORK["proxy_used"] is False, C.NETWORK


def test_the_proxy_is_only_for_an_upstream_only_wheel(one_pin, monkeypatch):
    """Tier 3: the mirror lacks torch, so the proxy is used -- for torch only."""
    proxy = one_pin / "set_proxy.sh"
    proxy.write_text("export http_proxy=http://127.0.0.1:17890\n")
    monkeypatch.setattr(C, "PROXY_SCRIPT", str(proxy))
    calls = []
    reads = {"n": 0}

    def fake_installed():
        reads["n"] += 1
        return {} if reads["n"] == 1 else {"torch": TORCH}

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        joined = " ".join(argv)
        if "download-r2.pytorch.org" in joined:
            return 0, "", ""
        if "download" in argv:
            return 0, "", ""
        return (1, "", "ERROR: No matching distribution found for torch==%s" % TORCH)

    monkeypatch.setattr(C, "_installed", fake_installed)
    monkeypatch.setattr(C, "_run", fake_run)

    rc, text = _run_main()
    report = _report(text)

    assert rc == 0, report
    overseas = [call for call in calls if "download-r2.pytorch.org" in " ".join(call)]
    assert overseas, "the upstream index was never tried: %r" % (calls,)
    assert all(call[0] == "bash" and "set_proxy.sh" in " ".join(call)
               for call in overseas), (
        "the overseas attempt must source the proxy: %r" % (overseas,))
    mirror = [call for call in calls if "mirrors.aliyun.com" in " ".join(call)]
    assert mirror and all("set_proxy.sh" not in " ".join(call) for call in mirror), (
        "the domestic mirror call must NOT use the proxy: %r" % (mirror,))
    assert C.NETWORK["proxy_used"] is True, C.NETWORK
    assert C.NETWORK["packages"] == ["torch"], C.NETWORK
    assert "upstream" in C.NETWORK["reason"], C.NETWORK


def test_a_mirror_missing_wheel_with_no_upstream_is_refused(one_pin, monkeypatch):
    """If the mirror lacks a wheel and no upstream index is declared, refuse by name."""
    monkeypatch.setattr(C, "OVERSEAS_INDEX", {})
    monkeypatch.setattr(C, "PROXY_SCRIPT", str(one_pin / "set_proxy.sh"))
    (one_pin / "set_proxy.sh").write_text(":\n")

    def fake_run(argv, **kwargs):
        return (1, "", "ERROR: No matching distribution found for torch==%s" % TORCH)

    monkeypatch.setattr(C, "_installed", lambda: {})
    monkeypatch.setattr(C, "_run", fake_run)

    rc, text = _run_main()
    report = _report(text)
    assert rc != 0
    assert report["code"] == "WHEEL_MISSING", report
    assert C.NETWORK["proxy_used"] is False, C.NETWORK


def test_only_one_reachable_findings_branch_remains():
    """C11: the second, unreachable `if args.findings` must be GONE.

    It sat after a branch that always returned, so it could never execute. Keeping
    it would make the file look like --findings was honoured twice while only one
    path was live.
    """
    source = (SCRIPTS / "configure.py").read_text()
    assert source.count("if args.findings:") == 1, (
        "configure.py still has more than one --findings branch; the trailing one "
        "is unreachable")
