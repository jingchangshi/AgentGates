"""MCP protocol tests: the real server, driven by a real MCP client over stdio.

These tests start python -m agent_gates.mcp_server as a subprocess and speak
the protocol to it. Nothing is mocked, because the properties under test -
schemas, structuredContent, error semantics and the bounded wait - exist only
in the protocol exchange.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_gates.manifest import digest

EXPECTED_TOOLS = [
    "cancel", "doctor", "list_gates", "list_workspaces", "report",
    "status", "submit", "verify", "verify_requirements", "wait",
]

PYTHON = sys.executable


CONFIG_TEMPLATE = """schema_version = 1
state_root = "{state}"
evidence_root = "{evidence}"

[adapters.test-pack]

[workspaces.ws]
source_root = "{source}"
adapters = ["test-pack"]

[worker]
backend = "systemd"
max_concurrent = 2
python = "{python}"
"""


class McpFixture(unittest.TestCase):
    """A workspace whose gate really executes a subprocess."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "input.txt").write_text("bytes")
        self.pack = self.root / "adapters/test-pack"
        (self.pack / "scripts").mkdir(parents=True)
        script = self.pack / "scripts/gate.py"
        script.write_text(
            "import json,sys\n"
            "if '--check-config' in sys.argv: sys.exit(0)\n"
            "print(json.dumps({'ok': True}))\n")
        manifest = {
            "schemaVersion": 1, "id": "test-pack", "version": "1",
            "files": {"scripts/gate.py": digest(script)},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": "test.run", "cwd": "source", "timeout": 60,
                         "argv": [PYTHON, "-I", "{pack}/scripts/gate.py"],
                         "configCheck": [PYTHON, "-I", "{pack}/scripts/gate.py",
                                         "--check-config"]}],
            "gates": [{"id": "test", "recipeId": "test.run",
                       "gateType": "deterministic-test",
                       "observationChecks": {"ok": True}}],
        }
        (self.pack / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "state").mkdir()
        (self.root / "evidence").mkdir()
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(CONFIG_TEMPLATE.format(
            state=self.root / "state",
            evidence=self.root / "evidence",
            source=self.source,
            python=PYTHON))
    def params(self):
        return StdioServerParameters(
            command=PYTHON,
            args=["-s", "-m", "agent_gates.mcp_server"],
            env={"AGENT_GATES_CONFIG": str(self.config_path),
                 "AGENT_GATES_ADAPTERS_ROOT": str(self.root / "adapters"),
                 "PATH": "/usr/bin:/bin",
                 "HOME": str(self.root)},
        )

    def call(self, action):
        """Run one coroutine against a live server session."""

        async def runner():
            async with stdio_client(self.params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await action(session)

        return anyio.run(runner)


class ProtocolTests(McpFixture):
    def test_all_tools_are_exposed_with_schemas(self):
        async def action(session):
            return (await session.list_tools()).tools

        tools = self.call(action)
        names = sorted(tool.name for tool in tools)
        self.assertEqual(names, EXPECTED_TOOLS)
        for tool in tools:
            self.assertIsNotNone(tool.input_schema, tool.name)
            self.assertIsNotNone(tool.output_schema, tool.name)
            self.assertEqual(tool.input_schema.get("type"), "object", tool.name)

    def test_list_workspaces_returns_structured_content(self):
        async def action(session):
            return await session.call_tool("list_workspaces", {})

        result = self.call(action)
        self.assertFalse(result.is_error)
        structured = result.structured_content
        self.assertIsNotNone(structured)
        self.assertEqual(structured["workspaces"][0]["id"], "ws")
        adapters = structured["workspaces"][0]["adapters"]
        self.assertEqual(adapters[0]["id"], "test-pack")
        self.assertEqual(adapters[0]["gates"][0]["id"], "test")

    def test_list_gates_returns_gate_metadata(self):
        async def action(session):
            return await session.call_tool(
                "list_gates", {"workspace": "ws", "adapter": "test-pack"})

        result = self.call(action)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["gates"][0]["id"], "test")
        self.assertEqual(result.structured_content["gates"][0]["checks"], {"ok": True})

    def test_unknown_workspace_is_an_error(self):
        async def action(session):
            return await session.call_tool("list_gates", {"workspace": "nope",
                                                        "adapter": "test-pack"})

        result = self.call(action)
        # A refusal must never look like a successful result.
        self.assertTrue(result.is_error)

    def test_undeclared_argument_is_refused(self):
        async def action(session):
            return await session.call_tool(
                "submit", {"workspace": "ws", "adapter": "test-pack",
                           "gates": ["test"], "shell_command": "rm -rf /"})

        result = self.call(action)
        self.assertTrue(result.is_error)
        self.assertIn("shell_command", str(result.content))

    def test_status_of_unknown_execution_is_an_error(self):
        async def action(session):
            return await session.call_tool("status", {"execution_id": "0" * 32})

        result = self.call(action)
        self.assertTrue(result.is_error)

    def test_doctor_reports_without_satisfying_requirements(self):
        async def action(session):
            return await session.call_tool("doctor", {"workspace": "ws",
                                                       "adapter": "test-pack"})

        result = self.call(action)
        self.assertEqual(result.structured_content["status"], "PASS")
        # A doctor PASS is not evidence for a run: there is no run id.
        async def verify_action(session):
            return await session.call_tool(
                "verify_requirements", {"workspace": "ws", "adapter": "test-pack",
                                        "required_gates": ["test"],
                                        "run_id": "0" * 32})

        verified = self.call(verify_action)
        self.assertFalse(verified.structured_content["satisfied"])


class DoctorScopeTests(McpFixture):
    """A doctor reply must state which route it speaks for.

    The fixture models the real shape of the problem: one adapter with a
    Docker route that cannot be configured here and a host route that can. An
    all-route report is BLOCKED_ENV because the Docker checks fail; a report
    scoped to the host route is PASS, and it must not be readable as a claim
    about Docker. Two recipes per route share one configuration check, which is
    what the fleet does when both routes read one machine file.
    """

    def setUp(self):
        super().setUp()
        script = self.pack / "scripts/gate.py"
        script.write_text(
            "import sys\n"
            "if '--check-config-docker' in sys.argv:\n"
            "    print('docker socket unreadable for this uid', file=sys.stderr)\n"
            "    sys.exit(2)\n"
            "if '--check-config' in sys.argv:\n"
            "    sys.exit(0)\n"
            "sys.exit(0)\n")
        # One shared check per route, on purpose: the fold must be reported, not
        # silently counted as a recipe that passed on its own.
        recipes = [
            ("two.docker-build", "--check-config-docker", []),
            ("two.docker-regression", "--check-config-docker", ["ascend-build"]),
            ("two.host-build", "--check-config", []),
            ("two.host-regression", "--check-config", ["ascend-build-host"]),
        ]
        manifest = {
            "schemaVersion": 1, "id": "test-pack", "version": "1",
            "files": {"scripts/gate.py": digest(script)},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": recipe_id, "cwd": "source", "timeout": 60,
                         "argv": [PYTHON, "-I", "{pack}/scripts/gate.py"],
                         "configCheck": [PYTHON, "-I", "{pack}/scripts/gate.py", flag]}
                        for recipe_id, flag, _ in recipes],
            "gates": [{"id": gate_id, "recipeId": recipe_id,
                       "gateType": "deterministic-test",
                       "observationChecks": {"exitCodeZero": True},
                       **({"dependsOn": depends} if depends else {})}
                      for (recipe_id, _, depends), gate_id in zip(
                          recipes, ["ascend-build", "ascend-regression",
                                    "ascend-build-host", "ascend-regression-host"])],
        }
        (self.pack / "manifest.json").write_text(json.dumps(manifest))

    def doctor(self, **arguments):
        async def action(session):
            return await session.call_tool("doctor", {"workspace": "ws",
                                                      "adapter": "test-pack", **arguments})
        return self.call(action)

    def test_unscoped_doctor_checks_every_route_and_never_folds(self):
        """Omitting the selection is the all-route diagnosis, unchanged."""
        structured = self.doctor().structured_content
        self.assertEqual(structured["status"], "BLOCKED_ENV")
        self.assertIs(structured["all_routes"], True)
        self.assertEqual(structured["selected_gates"],
                         ["ascend-build", "ascend-regression",
                          "ascend-build-host", "ascend-regression-host"])
        # Every recipe, in manifest order, each checked once - and no folding,
        # because a caller that saw fewer checks than recipes would lose the fact
        # that two scripts were consulted.
        self.assertEqual(structured["checked_recipes"],
                         ["two.docker-build", "two.docker-regression",
                          "two.host-build", "two.host-regression"])
        self.assertEqual(structured["deduplicated_recipes"], {})

    def test_scoped_doctor_reports_its_scope_and_excludes_other_routes(self):
        structured = self.doctor(gates=["ascend-build-host"]).structured_content
        self.assertEqual(structured["status"], "PASS")
        self.assertIs(structured["all_routes"], False)
        self.assertEqual(structured["selected_gates"], ["ascend-build-host"])
        self.assertEqual(structured["checked_recipes"], ["two.host-build"])
        # The unselected Docker route's check did not run at all: a scoped
        # diagnosis must not spend a subprocess on a route it cannot report on.
        self.assertNotIn("docker", json.dumps(structured["checked_recipes"]))

    def test_scoped_doctor_expands_dependencies_and_folds_one_check(self):
        structured = self.doctor(gates=["ascend-regression-host"]).structured_content
        self.assertEqual(structured["selected_gates"],
                         ["ascend-build-host", "ascend-regression-host"])
        # Both recipes declare one identical check, so it runs once and the fold
        # is stated rather than hidden.
        self.assertEqual(structured["checked_recipes"], ["two.host-build"])
        self.assertEqual(structured["deduplicated_recipes"],
                         {"two.host-regression": "two.host-build"})
        self.assertEqual(structured["status"], "PASS")

    def test_scoped_doctor_stays_blocked_when_the_selected_check_fails(self):
        """A passing route elsewhere must never turn a failing one into PASS."""
        structured = self.doctor(gates=["ascend-build"]).structured_content
        self.assertEqual(structured["status"], "BLOCKED_ENV")
        self.assertIs(structured["all_routes"], False)
        self.assertEqual(structured["selected_gates"], ["ascend-build"])
        self.assertEqual(structured["checked_recipes"], ["two.docker-build"])
        self.assertEqual(len(structured["failures"]), 1)

    def test_unknown_gate_is_refused(self):
        # The transport renders every handler exception as one opaque error, so
        # the assertion is the refusal itself; the readable reason is covered at
        # the engine boundary in test_config_and_api.
        result = self.doctor(gates=["no-such-gate"])
        self.assertTrue(result.is_error)


class LifecycleOverrideTests(McpFixture):
    """The gate machinery keeps working when the server is replaced."""

    def test_server_restart_does_not_disturb_records(self):
        async def doctor(session):
            return await session.call_tool("doctor", {"workspace": "ws",
                                                       "adapter": "test-pack"})

        first = self.call(doctor)
        self.assertEqual(first.structured_content["status"], "PASS")
        # A second, entirely separate server process must see the same state.
        second = self.call(doctor)
        self.assertEqual(second.structured_content["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
