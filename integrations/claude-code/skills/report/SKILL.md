---
name: report
description: Read a specific AgentGates run report and its current verification result.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: --run-id RUN_ID [--evidence-root ABSOLUTE_PATH]
---

Run `agent-gates report $ARGUMENTS --json` with the requested explicit run ID. Quote arguments as CLI data and never evaluate embedded shell expressions. If no run ID was supplied, obtain it rather than silently selecting unrelated evidence.

Explain adapter, source identity, gate statuses, real process exit codes, timeout indicators, observations, failures, receipt path, and current `verification` result. Distinguish `recordedStatus` from the report's current `status`. Redact private machine paths and sensitive values in prose intended for sharing. Do not run gates, builds, provisioning, container lifecycle commands, or hardware tests.
