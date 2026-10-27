# AgentGates V2 — systemd Worker Design

## The unit

One template, one instance per execution:

```
~/.config/systemd/user/agent-gates-worker@.service
agent-gates-worker@<executionId>.service
```

```ini
[Service]
Type=simple
ExecStart=<interpreter> -s -m agent_gates.worker %i
RuntimeMaxSec=<backstop>
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=30s
Restart=no
StandardOutput=journal
StandardError=journal
WorkingDirectory=/
Environment=PYTHONPATH=<package root>
Environment=AGENT_GATES_CONFIG=<config file>
Environment=AGENT_GATES_ADAPTERS_ROOT=<adapters dir>
Environment=<adapter-specific machine facts>
```

## Every decision, and why

### `Type=simple`, not `oneshot`

With `oneshot`, `systemctl start` **blocks until the gate finishes**, which would
make `submit()` synchronous — the exact opposite of the requirement. This was
measured: a 30-second gate made `submit` time out at 30s under `oneshot`, and
returned in 0.08s under `simple`.

`Restart=no` keeps the one-shot semantics: a cancelled or failed gate is not
retried behind the operator's back.

### `%i` is the only interpolation

The instance name is the only substitution, and `unit_for()` refuses anything
that is not `^[0-9a-f]{32}$` **before** a unit name is built. That is what stops
an execution id from becoming an argument to `systemctl`.

The interpreter, module entry point and configuration path are baked in by the
installer. No model input reaches `ExecStart`.

### `KillMode=control-group`

This is what makes `cancel()` mean "stop the compiler". A gate spawns a
supervisor, which spawns `docker exec`, which spawns `ninja`. Killing only the
worker's pid would leave the build running.

The resource leases are `flock` descriptors **held by those child processes**, so
they are released exactly when the tree is gone — not one moment earlier. That is
the mechanism behind "do not release a lock still in use".

### `RuntimeMaxSec` as the wedge backstop

A recipe owns its own timeout. `RuntimeMaxSec` is the second layer for a worker
whose own supervisor never returns. Measured: a real TileLang build gate and a
real regression gate both completed well inside it.

### Environment: fixed, not inherited

`systemd --user` starts the unit from the manager, not from the process that
submitted the request, so the caller's environment cannot leak in. The unit names
the exact configuration and package root.

`-s` (no user site) is used so a stray `~/.local` copy cannot shadow the
installed package. `-E`/`-I` are **not** used: they also ignore
`PYTHONPATH`, which is how the interpreter finds the package at all when the venv
is an editable install. This was found the hard way — the first three worker
starts failed with `ModuleNotFoundError` under `-I` and `-E`.

### Adapter-specific machine facts

An adapter that reads container identity from its process environment (as
`ascendnpu-ir` does) needs those values in the unit. `install()` takes an
`extra_env` mapping, written as `Environment=` lines. Names are validated against
`[A-Z_][A-Z0-9_]*` and values must be single-line, so a value cannot inject a new
unit directive.

## Lifecycle

```
submit()
  1. resolve the local configuration
  2. validate workspace / adapter / gate / device authorization
  3. check the concurrency limit
  4. persist ExecutionRequest  (durable BEFORE the unit starts)
  5. systemctl --user start agent-gates-worker@<id>.service
  6. return the execution id
```

Step 4 before step 5 is deliberate: a worker must never run a request the store
does not describe.

## Failure behaviour

| Condition | Result |
|---|---|
| no user systemd manager | `SYSTEMD_UNAVAILABLE`; **no in-process fallback** |
| unit start fails | `WORKER_START_FAILED`, execution `INTERRUPTED` |
| configuration changed since submit | worker refuses: `CONFIG_IDENTITY_CHANGED` |
| source lease held by another run | execution `FINISHED`, `RESOURCE_BUSY` |
| cancel before start | `CANCELLED`, build never starts |
| cancel during a gate | `CANCELLED`, gate FAIL receipt kept, child tree killed |
| worker dies, no receipt | `INTERRUPTED` |

## Installation

```python
from agent_gates import systemd_units
systemd_units.available()            # (usable, reason) - never raises
systemd_units.install(python, timeout_seconds=7200, extra_env={...})
```

`install()` writes the template, sets mode 0644 and runs `systemctl --user
daemon-reload`. It does not enable anything: each execution starts its own
instance.

## Diagnosing a worker

```bash
systemctl --user status "agent-gates-worker@<id>.service"
journalctl --user -u "agent-gates-worker@<id>.service" -n 50
```

The worker keeps a durable record beside its evidence:

```
state_root/executions/<id>/request.json    what was asked, and the config digest
state_root/executions/<id>/status.json     lifecycle state (terminal is final)
state_root/executions/<id>/events.jsonl    progress, diagnostic only
state_root/executions/<id>/worker.json     pid, boot id, interpreter
state_root/executions/<id>/cancel.json     present once a cancel was requested
```

`events.jsonl` is explicitly **not** evidence. A gate's PASS comes from the signed
receipt the Engine publishes, which the worker never writes itself.
