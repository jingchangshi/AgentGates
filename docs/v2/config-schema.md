# AgentGates V2 — Local Configuration Schema

Default location: `~/.config/agent-gates/config.toml`.

Override, in precedence order:
1. an explicit path passed to `config.load(path)`,
2. `$AGENT_GATES_CONFIG`,
3. the default above.

The worker unit records the resolved path at install time, so a running gate
always reads the configuration its request was bound to.

## Why this file exists

Objective C2 forbids an agent from naming any machine parameter. That rule is
enforced by *shape*: the tool input models simply have no field for a source
directory, an interpreter, a container or a device. Everything they cannot name
lives here, written by an operator.

## Full example

```toml
schema_version = 1

state_root    = "/home/agent/.local/state/agent-gates/v2"
evidence_root = "/home/agent/.local/state/agent-gates/v2/evidence"
# lock_root optional; defaults to <state_root>/locks

[adapters.tilelang-hivm]
# optional: an adapter that reads its toolchain from a file
env_file = "/home/agent/work/AgentGates/agent_gates/adapters/tilelang-hivm/env.sh"

[adapters.ascendnpu-ir]
# the same kind of file: container identity, build directory and toolchain
env_file = "/home/agent/work/AgentGates/agent_gates/adapters/ascendnpu-ir/env.sh"

[workspaces.tilelang]
source_root = "/home/agent/work/tilelang"
adapters = ["tilelang-hivm", "tilelang-hivm-npu"]
# optional allowlist; omit to permit every gate the adapter declares
allowed_gates = ["tilelang-hivm-build", "tilelang-hivm-regression"]

[worker]
backend = "systemd"            # the only backend; a fallback is not implemented on purpose
max_concurrent = 2
python = "/home/agent/work/AgentGates/.venv/bin/python"
start_timeout_seconds = 60

[devices]
# every device is usable by default; this section only narrows it
# allowed_npu_devices = [0, 1]
# require_exclusive_device = false
smi_command = "npu-smi"
```

## Field reference

| Field | Required | Rule |
|---|---|---|
| `schema_version` | yes | Must equal `1`. An unknown version is refused, not guessed. |
| `state_root` | yes | Absolute. Created if absent. Must not be a symlink or contain `..`. |
| `evidence_root` | yes | As above. Must be outside every workspace's source root (the engine enforces this too). |
| `lock_root` | no | Defaults to `<state_root>/locks`. One lock namespace per host user. |
| `workspaces.<id>` | yes, ≥1 | `id` must match `[a-z][a-z0-9-]{0,63}`. |
| `workspaces.<id>.source_root` | yes | Absolute, existing directory, no symlink components. |
| `workspaces.<id>.adapters` | yes | Non-empty list of adapter ids; each must have an `[adapters.<id>]` entry. |
| `workspaces.<id>.allowed_gates` | no | Non-empty list when present; a gate outside it is refused with `GATE_NOT_ALLOWED`. |
| `adapters.<id>.env_file` | no | Absolute, existing regular file, no symlink components. |
| `worker.backend` | yes | `systemd`. Any other value is refused. |
| `worker.max_concurrent` | no | Integer 1–64, default 2. A boolean is refused. |
| `worker.python` | no | Absolute, existing file. Defaults to the installing interpreter. |
| `worker.start_timeout_seconds` | no | Integer 1–3600, default 60. |
| `devices.allowed_npu_devices` | no | Integers 0-63 only. Absent or empty means every device. A quoted `"0"` is refused, because it would compare unequal to the engine's id. |
| `devices.smi_command` | no | Defaults to `npu-smi`. |

## Validation behaviour

Every refusal names the exact dotted field, because an operator who wrote the
file is the only one who can fix it:

```
ConfigError: workspaces.tilelang.source_root: parent component is a symlink: /home/agent/links
ConfigError: worker.max_concurrent: must be >= 1
ConfigError: devices.allowed_npu_devices[0]: must be an integer device id between 0 and 63
```

Paths are resolved to canonical absolute paths. Symlinked *components* are
refused, not followed: the recorded path must name the bytes the gate will use.

## The configuration digest

`config.config_digest` is a SHA-256 over the operational configuration: the
workspaces, their adapters and allowlists, the adapters' env files, the worker
settings and the authorized devices.

It deliberately **excludes** `state_root`, `evidence_root` and `lock_root`. Those
select where evidence is stored, not what a gate executes, so relocating evidence
must not invalidate every in-flight execution.

An execution binds the digest at submit time. If the file changes before the
worker starts, the worker refuses with `CONFIG_IDENTITY_CHANGED` rather than
silently adopting a different machine contract (objective C3).

## Device policy (ADR-008)

**Every NPU on the machine is available by default.** Sharing a card is normal:
a job already running on a device does not reserve it, and a gate that could not
start until an operator named a card would be blocked on a shared machine for no
technical reason.

```toml
# Optional. Omit the whole section and every device is usable, selecting 0.
[devices]
smi_command = "npu-smi"

# Optional restriction. Present and non-empty: only these may be used.
# allowed_npu_devices = [0, 1]

# Optional pin. Must appear in the allowlist when one is given.
# default_npu_device = 1

# Optional. true = refuse a card another process holds (BLOCKED_ENV).
# Use it for timing-sensitive or memory-heavy runs, not as a default.
# require_exclusive_device = false
```

| Field | Required | Rule |
|---|---|---|
| `allowed_npu_devices` | no | Integers 0–63. Absent or empty = every device. `authorized_npu_devices` is accepted as a legacy spelling. |
| `default_npu_device` | no | Integer 0–63; must also be allowed when an allowlist is present. |
| `require_exclusive_device` | no | Boolean, default false. |
| `smi_command` | no | Defaults to `npu-smi`. |

Selection is deterministic so two identical requests describe the same machine
state: `default_npu_device` if set, else the lowest allowed id, else `0`.

### Occupancy is reported, not enforced

The adapter still probes `npu-smi`, but to *describe* the card:

```
configuration ready: ... device=0 npuId=0 occupancy=shared(6) exclusive=False
```

A shared card runs. With `require_exclusive_device = true` the same situation
becomes `BLOCKED_ENV` (exit 75) instead. An unreadable driver answer is a refusal
in both cases, because "unknown" is not "shared".

**The tradeoff, stated plainly:** a shared card can make a timing-sensitive test
flaky, and that flakiness arrives as a verdict about the code. Read
`occupancy=shared(n)` before treating a failure as a regression, or opt into
exclusivity for those runs.

---
