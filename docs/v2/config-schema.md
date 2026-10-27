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
# no env_file: this adapter takes container identity from the worker environment

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
authorized_npu_devices = [1]   # empty means: no NPU gate may run
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
| `devices.authorized_npu_devices` | no | Integers only. A quoted `"0"` is refused, because it would compare unequal to the engine's normalised id. |
| `devices.smi_command` | no | Defaults to `npu-smi`. |

## Validation behaviour

Every refusal names the exact dotted field, because an operator who wrote the
file is the only one who can fix it:

```
ConfigError: workspaces.tilelang.source_root: parent component is a symlink: /home/agent/links
ConfigError: worker.max_concurrent: must be >= 1
ConfigError: devices.authorized_npu_devices[0]: must be a nonnegative integer device id
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

## Device authorization (C4)

An idle NPU is **not** an authorized NPU.

- `authorized_npu_devices = []` → any `-npu` adapter is refused with
  `DEVICE_NOT_AUTHORIZED`, and the test command is never invoked.
- Exactly one authorized id → that id is bound to the request.
- More than one authorized id → refused, because the manager will not choose a
  card on the operator's behalf.

Granting a device is an explicit, recorded operator decision. `npu-smi` is used
to *check* a device, never to *discover* a free one.
