"""Persisted P0 extraction audit references and deliberate behavior differences.

Audited source HEAD: 9ca04ba6814154a778ecf30168c793d73352e643.

Read engineering/{packs,runner,execution,preflight,provenance,workspace,
supervision,process_supervisor}.py, all three real adapter manifests, scripts
and READMEs, core runner/supervision/adapter tests, and project-adapter runtime
integration fixture. Tests never import the legacy runtime.

Retained contracts:
* manifest schemaVersion 1, SHA-pinned scripts, pack/source cwd, literal argv;
* stdout observation object limited to 65536 bytes for non-exit-only gates;
* nonzero exits, timeout and interruption cannot pass;
* source bytes before/after, Linux child-subreaper and inherited lease FDs;
* local HMAC protects data integrity without claiming same-user isolation.

Intentional corrections versus legacy behavior:
* JSON true requires a boolean, rather than accepting integer 1;
* duplicate JSON keys and NaN/Infinity are rejected;
* installed real adapters require a configuration precheck;
* receipts revalidate logs, manifest, current source and ignored build products;
* source index gitlink IDs and nested source bytes are included;
* each failed dependency explicitly prevents successor execution;
* historical PASS receipts are never reused.

Source symlinks bind link text without following arbitrary external targets.
Relevant executable products are separately resolved and hashed. Git-ignored
build outputs do not belong to the source snapshot. Docker inner process
lifetime is bounded by the adapter's fixed timeout; host Linux subreaping does
not grant ownership over remote/container processes.
"""
import unittest
from agent_gates.evaluator import equal


class AuditRegressionTests(unittest.TestCase):
    def test_legacy_boolean_equality_bug_is_deliberately_corrected(self):
        self.assertTrue(1 == True)
        self.assertFalse(equal(1, True))
