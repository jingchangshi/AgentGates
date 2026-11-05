"""The import path is ONE definition, and a run must not silently resolve a checkout.

The defect this file pins down: probe_env() excluded the tilelang source tree while
runtime_env() prepended it. The checks then verified the installed wheel while every
real run imported the checkout -- a green check that said nothing about what ran.

These are pure-function and single-subprocess tests: no container, no device.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
ENV_CONTEXT = PACK / "scripts" / "env_context.py"


def load_env_context():
    spec = importlib.util.spec_from_file_location("env_context_under_test", ENV_CONTEXT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def E():
    return load_env_context()


def split(path_value: str) -> list:
    return [entry for entry in path_value.split(os.pathsep) if entry]


def test_runtime_and_probe_agree_on_the_import_path(E) -> None:
    """The two environments must resolve tilelang the same way, by construction."""
    runtime = split(E.runtime_env(base={}, with_cann=False)["PYTHONPATH"])
    probe = split(E.probe_env(base={})["PYTHONPATH"])
    assert runtime == probe
    assert str(E.TILELANG_ROOT) not in runtime


def test_source_tree_is_absent_by_default(E) -> None:
    env = E.runtime_env(base={}, with_cann=False)
    assert str(E.TILELANG_ROOT) not in split(env["PYTHONPATH"])
    assert env[E.SOURCE_TREE_PREPENDED_ENV] == "0"


def test_source_tree_requires_an_explicit_switch_and_is_reported(E) -> None:
    env = E.runtime_env(base={}, with_cann=False, allow_source_tree=True)
    assert split(env["PYTHONPATH"])[0] == str(E.TILELANG_ROOT)
    assert env[E.SOURCE_TREE_PREPENDED_ENV] == "1"


def test_the_env_var_is_the_switch(E, monkeypatch) -> None:
    monkeypatch.setenv(E.ALLOW_SOURCE_TREE_ENV, "1")
    env = E.runtime_env(with_cann=False)
    assert split(env["PYTHONPATH"])[0] == str(E.TILELANG_ROOT)
    assert env[E.SOURCE_TREE_PREPENDED_ENV] == "1"


def test_probe_ignores_the_development_switch(E, monkeypatch) -> None:
    """The probe answers "what does the deployment resolve", so it never inherits it."""
    monkeypatch.setenv(E.ALLOW_SOURCE_TREE_ENV, "1")
    env = E.probe_env(base={})
    assert str(E.TILELANG_ROOT) not in split(env["PYTHONPATH"])
    assert env[E.SOURCE_TREE_PREPENDED_ENV] == "0"


def test_an_import_under_runtime_env_resolves_to_the_prefix(E) -> None:
    """The executable form of the rule: the actual import, not the path string."""
    env = E.runtime_env(base=dict(os.environ), with_cann=False)
    payload = E.assert_tilelang_from_prefix(env)
    assert payload["in_prefix"], payload
    assert payload["resolved"].startswith(str(E.SHARED_PREFIX))


def test_a_prefix_mismatch_is_refused_by_name(E, monkeypatch, tmp_path) -> None:
    """The assertion can fail: proving it only ever passes would prove nothing."""
    env = E.runtime_env(base=dict(os.environ), with_cann=False)
    with pytest.raises(E.ResolutionError) as excinfo:
        E.assert_tilelang_from_prefix(env, prefix=str(tmp_path / "not-the-prefix"))
    assert excinfo.value.code in ("TILELANG_NOT_FROM_PREFIX",
                                  "TILELANG_RESOLVES_TO_SOURCE_TREE")
