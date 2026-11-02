# ForgeKernel Ascend NPU adapter

This Adapter makes a ForgeKernel checkout runnable on an Ascend host and proves
that it is runnable. It is deliberately **not** a copy of the TileLang adapters:
ForgeKernel is an agent that searches kernels, and the toolchain it drives -
TileLang for Ascend - is a shared, already-built dependency. So this Adapter
ships host-local delivery, no Docker, no root assumption and no rebuild of
TileLang.

## Status: skeleton, and honest about it

| Step | Gate id | State |
| --- | --- | --- |
| preflight | `--check-config` / `--check-config-smoke` | implemented |
| npu smoke | `forgekernel-ascend-npu-smoke` | implemented |
| build | `forgekernel-ascend-npu-build` | implemented |
| optimize | — | **not implemented** |
| accept | — | **not implemented** |

The missing gates are absent rather than declared-and-always-passing. A gate
that exists and cannot fail is worse than one that does not exist: it would let a
run report a device result nobody measured. Adding one is a manifest edit plus a
real check, in that order.

## What the smoke gate proves

It answers one question: **can this run get the authorized card, and does the
card compute correct numbers?** Nothing more.

Inside a `deviceWindow` it allocates device tensors, runs elementwise `add`/`mul`
and a `sum` reduction in both FP32 and BF16, synchronizes, and compares against a
CPU reference. It publishes eight booleans, and the gate requires all eight:

- `npuAvailable`, `tensorOnDevice`, `deviceMemoryPositive` - a card was really
  reached and really allocated memory;
- `numericalOk` - FP32 within `1e-6`, BF16 within its own rounding floor
  (`2**-7`), and the reduction within that floor;
- `synchronizeOk` - the device reached a synchronisation point without error;
- `windowOpened`, `windowReleased`, `occupancyRecorded` - the lease was actually
  taken, actually released, and the occupancy it saw was a defined value.

It requires **only** `FK_ASCEND_PYTHON` plus a per-run device id. It does not
need TileLang, a ForgeKernel installation, a built wheel or a configured build
tree, so it can run before the build gate has ever succeeded.

Failure modes are separated on purpose:

| Situation | Reported as |
| --- | --- |
| `torch` or `torch_npu` will not import, or no device is present | `BLOCKED_ENV` (exit 75) - nothing was measured |
| the lease never opens within the window's wait | `BLOCKED_ENV` (exit 75) - our own contention, nothing ran |
| the device ran but the numbers are wrong | `FAIL` - a real defect |
| the probe printed no readable report | `INVALID_EVIDENCE` - a claim without evidence |

### The window covers work, not setup

Measured on the target machine, the phases before any kernel runs are expensive
and none of them touch hardware:

| Phase | Cost | Inside the window? |
| --- | ---: | --- |
| `import torch` | ~0.6 s | no |
| `import torch_npu` | ~1.2 s | no |
| device context creation | ~21 s | no |
| occupancy probe (`npu-smi`) | ~0.07 s | at the window's edge |
| allocation + arithmetic + synchronize | **~0.07 s** | **yes** |

So the probe warms up first and takes the lease afterwards, and the whole
window is a fraction of a second instead of twenty. A window dominated by import
time would be exactly the "occupying a card while not using it" this design
exists to prevent, and it would let a receipt report a 20-second "device time"
that measured an `import`.

The receipt carries the split - `warmupSeconds` outside the window, and
`heldSeconds` / `probeSeconds` inside it - so a reader can check the claim rather
than trust it. These durations are **recorded, never gated**: a slow machine is
not a failed test, and a time bound would make the verdict depend on what else
the host was doing.

## What the build gate proves

It installs the adapted ForgeKernel tree into a private virtual environment and
then proves the installed package is real:

* `venvCreated` - the private environment exists and has an interpreter;
* `forgekernelInstalled` - `pip install` succeeded;
* `importOk` - the installed `forgekernel` imports in a fresh `-I` interpreter;
* `versionReported` - it reports a nonempty version;
* `forgeEntryPointOk` - the CLI module the console script points at exists;
* `tilelangReused` - the reused TileLang checkout is the same one before and
  after the install, re-read rather than assumed;
* `tilelangSourceUnchanged` - the source tree it installed from was not edited.

Missing data is false, never assumed. A wheel that installs but cannot import
fails, and an occupied device does not: this gate declares `"device": "none"`,
because building never touches a card.

## The device lease is a window, not a reservation

This machine is shared. Every card is in use by other people most of the time,
and that is **not** an error condition: refusing to run because a card is busy
would block legitimate work for a reason unrelated to the code under test.

So a device lease here is a **window**, not a reservation:

* the engine does **not** hold `npu:<id>` for the whole recipe; a recipe that
  declares `deviceWindow` opens the lease itself;
* the window covers only the work that actually drives the card. Thinking,
  compiling and writing files happen outside it, so a long optimization turn
  never holds hardware it is not using;
* the window **waits** for a bounded time instead of failing immediately,
  because two of our own runs wanting one card a second apart is ordinary
  coordination, not an environment fault;
* the window reports what it saw - which lease, when it opened, when it closed,
  how long it waited, and the observed occupancy (`free`, `shared(n)`,
  `unknown`). Occupancy is **recorded, never gated**.

What a lease does not do: it coordinates AgentGates runs **with each other**. It
cannot and does not exclude other tenants of the machine, and no claim here
should be read as one.

A window that never opens exits `75`, which the runner records as `BLOCKED_ENV` -
a resource refusal where nothing ran, not a failed test. An occupied card never
produces that: only our own contention can.

### Engine side

`agent_gates/engine.py` behaves unchanged unless a manifest opts in:

* no `deviceWindow` in a recipe -> the engine takes the whole-recipe `npu:<id>`
  lease exactly as it always has;
* `deviceWindow: {"waitSeconds": N}` -> the engine passes the lease path, the
  resource name and the wait bound in `AGENT_GATES_DEVICE_LEASE_*`, and holds
  nothing itself;
* a manifest that mixes both kinds of device recipe in one adapter is refused,
  because the two modes cannot share one run without deadlocking on the very
  lease the child would wait for;
* the receipt gains `deviceLease` (mode, resource, gates) and, per gate, the
  `deviceWindow` the child reported. A windowed gate whose report names a
  different lease, has end before start, or carries an undefined occupancy value
  is `INVALID_EVIDENCE`.

## Configuration

One strict `NAME=value` file, `env.sh`, beside `env.example.sh`. Pass an absolute
path with `--env-file` (mapped to `FK_ASCEND_ENV_FILE`) or keep the git-ignored
`env.sh` in place. The Gate parses the file as data and never executes it.
Unknown keys, duplicate keys and malformed lines fail.

`FK_ASCEND_DEVICE_ID` is deliberately not in that file: it authorizes one run,
not a machine. The provider API key is never in the file either - the file names
the environment variable that holds it.

## Why the source is never written

`sourceWrite` is false and `FK_ASCEND_VENV` lives under `FK_ASCEND_WORK_ROOT`,
outside both the checkout and the evidence directory. The build reads the tree,
installs into the private environment, and re-reads the identity it recorded so a
concurrent edit cannot leave a receipt describing a tree that no longer exists.

## Reusing TileLang is a dependency, not a build step

`FK_ASCEND_TILELANG_SOURCE` and `FK_ASCEND_TILELANG_BUILD` name an existing
checkout and an existing build. This Adapter records their identity - module
digest and every `libtilelang*.so` digest - and never configures, compiles or
`pip install`s TileLang. If the native library is missing the Gate stops with a
diagnosis rather than silently beginning a build nobody approved. Reusing a build
is only honest while it is pinned: a change to that checkout invalidates the
evidence and needs a new doctor/build run.

## Commands

```bash
# preflight only: no build, no device, no network
agent-gates doctor --adapter forgekernel-ascend-npu --source /absolute/fk --env-file /absolute/env.sh --json

# the build gate
agent-gates run --adapter forgekernel-ascend-npu --source /absolute/fk --env-file /absolute/env.sh --all --json
```

See [adapter configuration](../../../../docs/adapters.md) for the general
adapter, environment-file and receipt conventions.
