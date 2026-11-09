"""Configuration and versioned-API contract tests.

Every assertion here is about a refusal or an identity: which field a bad file
names, which path a value resolves to, and which digest two configurations
share. Those are the properties the agent-facing surface depends on.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from agent_gates import api_types as api
from agent_gates import config as cfg
from agent_gates.cli import main
from agent_gates.engine import Engine
from agent_gates.evidence import EvidenceStore
from agent_gates.manifest import digest
from agent_gates.registry import Registry


def write_config(tmp_path: Path, body: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


@pytest.fixture
def machine(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "evidence").mkdir()
    (tmp_path / "source").mkdir()
    (tmp_path / "env.sh").write_text("export A=1\n")
    return tmp_path


def minimal(machine: Path, *, extra: str = "") -> str:
    return f"""
schema_version = 1
state_root = "{machine / 'state'}"
evidence_root = "{machine / 'evidence'}"

[adapters.ascendnpu-ir]
env_file = "{machine / 'env.sh'}"

[workspaces.ascend]
source_root = "{machine / 'source'}"
adapters = ["ascendnpu-ir"]

[worker]
backend = "systemd"
max_concurrent = 2
python = "{sys.executable}"
{extra}
"""


def load_text(machine, text):
    return cfg.from_mapping(cfg.tomllib.loads(text), path=machine / "config.toml")


# ── happy path ───────────────────────────────────────────────────────────────

def test_minimal_config_loads_and_normalises_paths(machine):
    config = load_text(machine, minimal(machine))
    assert config.schema_version == cfg.CONFIG_SCHEMA_VERSION
    assert config.state_root == (machine / "state").resolve()
    assert config.evidence_root == (machine / "evidence").resolve()
    assert config.workspaces["ascend"].source_root == (machine / "source").resolve()
    assert config.workspaces["ascend"].adapters == ("ascendnpu-ir",)
    assert config.adapters["ascendnpu-ir"].env_file == (machine / "env.sh").resolve()
    assert config.worker.backend == "systemd"
    assert config.worker.max_concurrent == 2


def test_digest_is_stable_across_equivalent_files(machine):
    first = load_text(machine, minimal(machine))
    second = load_text(machine, minimal(machine))
    assert first.config_digest == second.config_digest
    assert len(first.config_digest) == 64


def test_digest_changes_when_operational_value_changes(machine):
    base = load_text(machine, minimal(machine))
    changed = load_text(machine, minimal(machine).replace("max_concurrent = 2", "max_concurrent = 3"))
    assert base.config_digest != changed.config_digest


def test_digest_ignores_evidence_and_state_location(machine):
    """Moving evidence storage must not invalidate an in-flight execution."""
    base = load_text(machine, minimal(machine))
    (machine / "state2").mkdir()
    moved = load_text(machine, minimal(machine).replace(
        f'state_root = "{machine / "state"}"', f'state_root = "{machine / "state2"}"'))
    assert base.config_digest == moved.config_digest


def test_allowlist_parsed(machine):
    text = minimal(machine).replace(
        'adapters = ["ascendnpu-ir"]',
        'adapters = ["ascendnpu-ir"]\nallowed_gates = ["ascend-build"]')
    config = load_text(machine, text)
    assert config.workspaces["ascend"].allowed_gates == {"ascend-build"}


# ── negative: each names its field ───────────────────────────────────────────

def test_unsupported_schema_version_names_field(machine):
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, minimal(machine).replace("schema_version = 1", "schema_version = 9"))
    assert caught.value.field == "schema_version"
    assert "unsupported" in caught.value.reason


def test_missing_state_root_names_field(machine):
    text = minimal(machine).replace(f'state_root = "{machine / "state"}"', "")
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "state_root"


def test_relative_path_names_field(machine):
    text = minimal(machine).replace(f'state_root = "{machine / "state"}"', 'state_root = "state"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "state_root"
    assert "absolute" in caught.value.reason


def test_dotdot_escape_rejected(machine):
    text = minimal(machine).replace(f'state_root = "{machine / "state"}"', 'state_root = "/tmp/../etc"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "state_root"
    assert ".." in caught.value.reason


def test_missing_directory_rejected(machine):
    text = minimal(machine).replace(f'"{machine / "source"}"', f'"{machine / "nope"}"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "workspaces.ascend.source_root"
    assert "does not exist" in caught.value.reason


def test_symlinked_source_rejected(machine):
    link = machine / "link"
    link.symlink_to(machine / "source")
    text = minimal(machine).replace(f'"{machine / "source"}"', f'"{link}"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "workspaces.ascend.source_root"
    assert "symlink" in caught.value.reason


def test_symlinked_env_file_rejected(machine):
    link = machine / "env-link.sh"
    link.symlink_to(machine / "env.sh")
    text = minimal(machine).replace(f'"{machine / "env.sh"}"', f'"{link}"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "adapters.ascendnpu-ir.env_file"
    assert "symlink" in caught.value.reason


def test_workspace_naming_unconfigured_adapter_rejected(machine):
    text = minimal(machine).replace('adapters = ["ascendnpu-ir"]', 'adapters = ["tilelang-hivm"]')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "workspaces.ascend.adapters"
    assert "tilelang-hivm" in caught.value.reason


def test_invalid_workspace_id_names_field(machine):
    # A quoted key parses as TOML, so this reaches our identifier rule rather
    # than dying in the parser.
    body = minimal(machine).replace("[workspaces.ascend]", '[workspaces."Ascend_1"]')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, body)
    assert caught.value.field == "workspaces.Ascend_1"
    assert "workspace id" in caught.value.reason


def test_workspace_id_with_path_separator_rejected(machine):
    """An id is joined onto directories later; a separator must never survive."""
    body = minimal(machine).replace("[workspaces.ascend]", '[workspaces."../etc"]')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, body)
    assert "workspace id" in caught.value.reason


def test_unknown_worker_backend_rejected(machine):
    text = minimal(machine).replace('backend = "systemd"', 'backend = "threads"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "worker.backend"


def test_boolean_concurrency_rejected(machine):
    """TOML true is a Python int; refusing it keeps "concurrency = true" from meaning 1."""
    text = minimal(machine).replace("max_concurrent = 2", "max_concurrent = true")
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "worker.max_concurrent"


def test_concurrency_bounds_enforced(machine):
    text = minimal(machine).replace("max_concurrent = 2", "max_concurrent = 0")
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "worker.max_concurrent"


def test_missing_python_rejected(machine):
    text = minimal(machine).replace(f'python = "{sys.executable}"', 'python = "/nonexistent/python3"')
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "worker.python"


# ── NPU device policy (ADR-008) ─────────────────────────────────────────────

def test_every_device_is_available_by_default(machine):
    """No [devices] section at all means every card may be used.

    Sharing an NPU is normal: a job already on a card does not reserve it. A
    gate that could not run until an operator named a card would be blocked on
    a shared machine for no technical reason.
    """
    config = load_text(machine, minimal(machine))
    assert config.devices.allowed is None
    assert config.devices.permits(0) and config.devices.permits(7)
    assert config.devices.require_exclusive is False


def test_empty_allowlist_also_means_every_device(machine):
    """An existing file with an empty list keeps loading and stays permissive."""
    text = minimal(machine) + "\n[devices]\nauthorized_npu_devices = []\n"
    config = load_text(machine, text)
    assert config.devices.allowed is None
    assert config.devices.permits(3)


def test_allowlist_narrows_the_set(machine):
    text = minimal(machine) + "\n[devices]\nallowed_npu_devices = [0, 3]\n"
    config = load_text(machine, text)
    assert config.devices.allowed == {"0", "3"}
    assert config.devices.permits(3) is True
    assert config.devices.permits(1) is False


def test_selection_is_deterministic(machine):
    """Two identical requests must not land on different hardware."""
    default = load_text(machine, minimal(machine))
    assert default.devices.select() == "0"
    narrowed = load_text(machine, minimal(machine) + "\n[devices]\nallowed_npu_devices = [5, 2]\n")
    assert narrowed.devices.select() == "2"
    pinned = load_text(machine, minimal(machine) + "\n[devices]\ndefault_npu_device = 5\n")
    assert pinned.devices.select() == "5"


def test_default_device_must_be_allowed(machine):
    text = (minimal(machine) + "\n[devices]\nallowed_npu_devices = [1]\n"
            "default_npu_device = 2\n")
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "devices.default_npu_device"


def test_string_device_id_rejected(machine):
    """A quoted id would compare unequal to the engine normalised id."""
    text = minimal(machine) + "\n[devices]\nallowed_npu_devices = ['0']\n"
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "devices.allowed_npu_devices[0]"


def test_exclusive_device_is_opt_in(machine):
    default = load_text(machine, minimal(machine))
    assert default.devices.require_exclusive is False
    exclusive = load_text(machine, minimal(machine)
                          + "\n[devices]\nrequire_exclusive_device = true\n")
    assert exclusive.devices.require_exclusive is True


def test_exclusive_flag_must_be_a_boolean(machine):
    text = minimal(machine) + "\n[devices]\nrequire_exclusive_device = 'yes'\n"
    with pytest.raises(cfg.ConfigError) as caught:
        load_text(machine, text)
    assert caught.value.field == "devices.require_exclusive_device"


def test_device_policy_changes_the_digest(machine):
    base = load_text(machine, minimal(machine))
    narrowed = load_text(machine, minimal(machine)
                         + "\n[devices]\nallowed_npu_devices = [0]\n")
    exclusive = load_text(machine, minimal(machine)
                          + "\n[devices]\nrequire_exclusive_device = true\n")
    assert base.config_digest != narrowed.config_digest
    assert base.config_digest != exclusive.config_digest

# ── file loading ─────────────────────────────────────────────────────────────

def test_load_from_path(machine):
    path = write_config(machine, minimal(machine))
    config = cfg.load(path)
    assert config.path == path
    assert config.workspaces["ascend"].id == "ascend"


def test_load_uses_environment_override(machine, monkeypatch):
    path = write_config(machine, minimal(machine))
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(path))
    assert cfg.load().workspaces["ascend"].source_root == (machine / "source").resolve()


def test_missing_file_is_a_config_error(machine):
    with pytest.raises(cfg.ConfigError) as caught:
        cfg.load(machine / "absent.toml")
    assert caught.value.field == "config"


def test_invalid_toml_is_a_config_error(machine):
    path = write_config(machine, "schema_version = ")
    with pytest.raises(cfg.ConfigError) as caught:
        cfg.load(path)
    assert caught.value.field == "config"
    assert "TOML" in caught.value.reason


def test_config_file_symlink_rejected(machine):
    real = write_config(machine, minimal(machine))
    link = machine / "link.toml"
    link.symlink_to(real)
    with pytest.raises(cfg.ConfigError) as caught:
        cfg.load(link)
    assert "symlink" in caught.value.reason


def test_accessors_reject_unknown_ids(machine):
    config = load_text(machine, minimal(machine))
    with pytest.raises(cfg.ConfigError) as caught:
        config.workspace("nope")
    assert caught.value.field == "workspaces.nope"
    with pytest.raises(cfg.ConfigError) as caught:
        config.adapter("nope")
    assert caught.value.field == "adapters.nope"


# ── API contract ─────────────────────────────────────────────────────────────

def test_error_family_mapping_is_total():
    """Every declared code must resolve to a family, so no error is unclassified."""
    for code in api.ErrorCode:
        assert code in api.FAMILY_OF_CODE


def test_error_carries_code_family_and_serialises():
    error = api.APIError(api.ErrorCode.RESOURCE_BUSY, "source busy", field="source", details={"path": "/x"})
    assert error.family is api.ErrorFamily.RESOURCE
    model = error.to_model()
    assert api.jsonable(model) == {
        "code": "RESOURCE_BUSY", "family": "resource", "message": "source busy",
        "field": "source", "details": {"path": "/x"},
    }


def test_blocked_env_is_a_gate_status_not_an_error_code():
    """BLOCKED_ENV describes a gate verdict; it must never be an API error code."""
    assert "BLOCKED_ENV" not in {code.value for code in api.ErrorCode}
    assert api.GateStatus.BLOCKED_ENV.value == "BLOCKED_ENV"


def test_execution_models_are_closed():
    with pytest.raises(Exception):
        api.ExecutionRequest(workspace="w", adapter="a", gates=["g"], shell_command="rm -rf /")


def test_execution_request_has_no_machine_parameter_fields():
    """C2: the tool-facing request must not be able to carry machine settings."""
    fields = set(api.ExecutionRequest.model_fields)
    assert fields == {"workspace", "adapter", "gates", "request_id"}


def test_terminal_states_are_exactly_the_three_outcomes():
    assert api.TERMINAL_STATES == {api.ExecutionState.FINISHED,
                                   api.ExecutionState.CANCELLED,
                                   api.ExecutionState.INTERRUPTED}


def test_finished_status_is_not_a_pass():
    """FINISHED describes the worker stopping, not the gates passing."""
    status = api.ExecutionStatus(execution_id="e" * 32, state=api.ExecutionState.FINISHED,
                                 workspace="w", adapter="a", gates=["g"],
                                 run_id="r" * 32, exit_reason="completed")
    assert "FAIL" not in api.jsonable(status)["state"]
    assert api.jsonable(status)["state"] == "FINISHED"


def test_requirement_verification_result_defaults_are_safe():
    result = api.RequirementVerificationResult(
        satisfied=False, status=api.GateStatus.NOT_RUN, verified_at="t",
        workspace="w", adapter="a")
    assert result.run_id is None
    assert result.matched_gates == []
    assert result.failures == []


# ── doctor scope (D8) ────────────────────────────────────────────────────────

def test_doctor_report_defaults_to_the_all_route_claim():
    """A report that says nothing about scope is the historical all-route one.

    The default matters more than it looks: if the narrower claim were the
    default, a payload from an older producer would read as scoped and an
    operator could quote a host-only PASS where a full diagnosis was meant.
    """
    report = api.DoctorReport(workspace="w", adapter="a", status=api.GateStatus.PASS)
    assert report.all_routes is True
    assert report.selected_gates == []
    assert report.checked_recipes == []
    assert report.deduplicated_recipes == {}


def test_doctor_report_states_a_scoped_claim_explicitly():
    report = api.DoctorReport(
        workspace="w", adapter="a", status=api.GateStatus.PASS,
        all_routes=False, selected_gates=["host-regression"],
        checked_recipes=["host-build"],
        deduplicated_recipes={"host-regression": "host-build"})
    rendered = api.jsonable(report)
    assert rendered["all_routes"] is False
    assert rendered["selected_gates"] == ["host-regression"]
    assert rendered["deduplicated_recipes"] == {"host-regression": "host-build"}


def test_doctor_report_rejects_an_undeclared_scope_key():
    """The model stays closed, so no surface grows a private scope field."""
    with pytest.raises(Exception):
        api.DoctorReport(workspace="w", adapter="a", status=api.GateStatus.PASS,
                         scope={"allRoutes": True})


# ── doctor scoping: engine and CLI (D8) ──────────────────────────────────────

@pytest.fixture
def two_routes(tmp_path):
    """One adapter, two routes: Docker checks fail here, host checks pass.

    This is the machine the feature exists for. The two recipes of each route
    share one configuration check, exactly as the real adapter does when both
    routes read the same machine file, so the fold is exercised rather than
    described.
    """
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "evidence").mkdir()
    pack = tmp_path / "adapters/two-route"
    (pack / "scripts").mkdir(parents=True)
    script = pack / "scripts/gate.py"
    script.write_text(
        "import sys\n"
        "if '--check-config-docker' in sys.argv:\n"
        "    print('docker socket unreadable for this uid', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "if '--check-config' in sys.argv:\n"
        "    sys.exit(0)\n"
        "sys.exit(0)\n")
    routes = [("docker-build", "--check-config-docker", None),
              ("docker-regression", "--check-config-docker", "docker-build"),
              ("host-build", "--check-config", None),
              ("host-regression", "--check-config", "host-build")]
    (pack / "manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "id": "two-route", "version": "1",
        "files": {"scripts/gate.py": digest(script)},
        "permissions": {"sourceWrite": False, "hostExecution": True},
        "recipes": [{"id": f"two.{gate_id}", "cwd": "source", "timeout": 60,
                     "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
                     "configCheck": [sys.executable, "-I", "{pack}/scripts/gate.py", flag]}
                    for gate_id, flag, _ in routes],
        "gates": [{"id": gate_id, "recipeId": f"two.{gate_id}",
                   "gateType": "deterministic-test",
                   "observationChecks": {"exitCodeZero": True},
                   **({"dependsOn": [depends]} if depends else {})}
                  for gate_id, _, depends in routes],
    }))
    engine = Engine(registry=Registry(tmp_path / "adapters"),
                    store=EvidenceStore(tmp_path / "evidence"))
    return engine, source


def test_unscoped_doctor_keeps_the_all_route_behaviour(two_routes):
    """Omitting the selection is today's behaviour: every recipe, aggregated."""
    engine, source = two_routes
    result = engine.doctor("two-route", source)
    assert result["allRoutes"] is True
    assert result["selectedGates"] == ["docker-build", "docker-regression",
                                       "host-build", "host-regression"]
    assert result["checkedRecipes"] == ["two.docker-build", "two.docker-regression",
                                        "two.host-build", "two.host-regression"]
    # No folding in the unscoped path: an existing caller must see the same
    # number of checks it always saw.
    assert result["deduplicatedRecipes"] == {}
    assert result["status"] == "BLOCKED_ENV"


def test_scoped_doctor_reports_scope_and_never_claims_other_routes(two_routes):
    engine, source = two_routes
    result = engine.doctor("two-route", source, selected=["host-build"])
    assert result["status"] == "PASS"
    assert result["allRoutes"] is False
    assert result["selectedGates"] == ["host-build"]
    assert result["checkedRecipes"] == ["two.host-build"]


def test_scoped_doctor_expands_dependencies_and_folds_one_check(two_routes):
    engine, source = two_routes
    result = engine.doctor("two-route", source, selected=["host-regression"])
    assert result["selectedGates"] == ["host-build", "host-regression"]
    assert result["checkedRecipes"] == ["two.host-build"]
    assert result["deduplicatedRecipes"] == {"two.host-regression": "two.host-build"}
    assert result["status"] == "PASS"


def test_scoped_doctor_is_still_blocked_when_the_selected_check_fails(two_routes):
    """Never PASS because a route the caller did not select happens to pass."""
    engine, source = two_routes
    result = engine.doctor("two-route", source, selected=["docker-build"])
    assert result["status"] == "BLOCKED_ENV"
    assert result["allRoutes"] is False
    assert result["checkedRecipes"] == ["two.docker-build"]
    assert len(result["failures"]) == 1


def test_unknown_gate_is_refused_before_a_run_directory_is_created(two_routes, tmp_path):
    """A typo costs an error, not an evidence tree to clean up."""
    engine, source = two_routes
    doomed = source.parent / "evidence" / "runs"
    before = sorted(doomed.iterdir()) if doomed.is_dir() else []
    with pytest.raises(ValueError, match="unknown gate"):
        engine.doctor("two-route", source, selected=["host-bulid"])
    after = sorted(doomed.iterdir()) if doomed.is_dir() else []
    assert after == before


def test_doctor_scope_survives_the_empty_selection(two_routes):
    """An empty list is the all-route default, never "check nothing"."""
    engine, source = two_routes
    result = engine.doctor("two-route", source, selected=[])
    assert result["allRoutes"] is True
    assert len(result["checkedRecipes"]) == 4


def test_cli_doctor_refuses_an_unknown_gate(tmp_path):
    """The flag is real on the doctor subcommand, and a bad id exits 2.

    The refusal happens before a run directory exists, so the CLI exit code is
    the caller-misuse code 2 rather than the BLOCKED_ENV diagnosis code 1.
    """
    adapter = Path(__file__).resolve().parent.parent / "agent_gates/adapters/ascendnpu-ir"
    if not (adapter / "manifest.json").is_file():
        pytest.skip("ascendnpu-ir adapter is not installed in this checkout")
    # The evidence directory is a sibling of the source root, never inside it:
    # the engine refuses that before any check runs, which is a different test.
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "evidence"
    refused = main(["doctor", "--adapter", "ascendnpu-ir", "--source", str(source),
                    "--gate", "no-such-gate", "--evidence-root", str(evidence), "--json"])
    assert refused == 2
