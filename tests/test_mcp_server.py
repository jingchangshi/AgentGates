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
