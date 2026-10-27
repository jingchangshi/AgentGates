---
name: doctor
description: Check an explicitly selected AgentGates adapter and source configuration without running compiler or device gates.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: --adapter ID --source ABSOLUTE_PATH [--env-file ABSOLUTE_PATH] [--device-id ASSIGNED_ID]
---

Run the installed standalone CLI using the user's explicit arguments:

`agent-gates doctor $ARGUMENTS --json`

Treat arguments as CLI data; quote paths and never evaluate shell expressions embedded in them. If the adapter or source is absent, obtain those values before running. A device ID must be explicitly assigned by the user. Do not discover or substitute a device. Report the actual status, failures, and configuration check result. Doctor checks configuration; it does not compile, run tests, provision environments, create containers, or touch NPU hardware. Do not follow it with `run` automatically.
