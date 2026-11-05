# tilefoundry-npu

The TileFoundry-on-Ascend Adapter: build, CPU regression and device regression.

## The one rule

**TileFoundry work happens inside the container.** There is no host path.

That is not a style preference. A host run and a container run are different
claims about the same code, and an evidence record that does not say which
environment produced it cannot be checked by anyone later. So `--cpu` and
`--device` call `assert_not_host()` before they touch the toolchain: on a host
they refuse with exit 2 and print the `docker exec` that would have worked.

The check is `/.dockerenv`, which the runtime creates. A marker this code
exported itself would be satisfied by the act of exporting it.

## One environment, and why it lives under /home

```
/home/s00653124/.conda/envs/tilefoundry-npu/bin/python   (Python 3.12.16)
```

A **conda** environment containing torch 2.12.0+cpu, torch_npu 2.12.0, the
TileFoundry dependencies and z3. It lives under `/home`, which is bind-mounted,
so it survives container recreation with no reinstall and no re-download -- and a
rebuild costs about an hour at direct network speed.

There used to be two environments, split by Python version, because every local
`torch_npu` wheel was `cp311` and paired with `torch 2.10.0+cpu`. That constraint
is gone: `torch_npu 2.12.0` exists as a `cp312` build matching `torch 2.12.0+cpu`,
so one 3.12 environment serves both CPU and device work.

`/opt/tf-venv` is **deprecated** and no longer created or mounted. The old
`tfpy312` checkout is not used as an interpreter at all.

### CANN must be sourced first

`. /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh` must run **before** `import
torch_npu`, or it fails with:

```
ImportError: libhccl.so: cannot open shared object file: No such file or directory
```

`torch_npu._C` resolves native libraries against the CANN layout that `set_env.sh`
puts on the loader path. The order is fixed: CANN, then conda activate, then
`ASCEND_RT_VISIBLE_DEVICES`, then import. Two traps: the script breaks under
`set -u`, and the path has an `ascend-toolkit/` component that is easy to omit.

### A shared card is entered, not refused

All Ascend cards are shared. Before anything imports `torch_npu` or reaches
`aclInit`, the device Gate reads occupancy once and records it. It then runs the
same workload whether `occupied` is true or false. **Busy is metadata, never a
dispatch decision.**

This section replaces an earlier policy that refused a held card by default. That
policy and its `TF_ALLOW_BUSY_DEVICE` override are **retired: do not restore them.**
No current instruction in this adapter asks an operator to set a busy override,
and the legacy keys are accepted only so old configuration files parse; they
cannot change dispatch, selection, verdict, or a retry.

The occupancy read stays **above** the probe only so a real health/readability
fault is reported before a device node is opened. It is not an admission test and
must not acquire one.

### Network

Direct access is ~58 KB/s. Through the host proxy it is ~1.7 MB/s, about 30x, and
torch alone is 184 MB. `setup-tf-env.sh` reads the proxy exports from `PROXY_SH`
and injects them into `docker exec` with `-e`.

## What this evidence is, and is not

The Gate writes a local JSON report. That report is NOT a verified service receipt,
and this adapter is NOT registered with the AgentGates service.

- list_workspaces on the agentgates server returns the workspaces ascendnpu-ir and
  tilelang. It does NOT list tilefoundry-npu. There is no server-side gate record for
  this adapter, and no run id exists for it.
- Consequently the JSON under --evidence-dir is this process describing itself. It is
  inspectable evidence that a run happened and what it observed, but it carries no
  independent attestation. Treat it as a local artifact signed by nobody, not a receipt.

What the CPU Gate DOES now guarantee, because each item below was broken once and is
now checked from evidence rather than from an exit code:

  interpreter identity challenge   refuses a config pointing IN_CONTAINER_PYTHON at
                                   /bin/true (the reproduced FALSE PASS: exit 0, every
                                   field true, nothing run)
  structured per-tool reports      refuses a tool that exits 0 but emits no PASS report,
                                   or a report whose status/exit/checks disagree
  typed finalized artifact         refuses finalize exiting 0 without producing
                                   parseable TIR (hash and size recorded)
  actual in-container uid          refuses ran_as being a configured string; the uid is
                                   read with id -u and must be non-root and equal to
                                   CONTAINER_UID

Both the identity and the report checks have NEGATIVE tests that prove the refusal
fires (tests/identity_check.py, tests/tool_report_check.py), because a check that
cannot fail is the exact defect this section exists to prevent.

An M3 acceptance receipt produced elsewhere in this session remains valid on its own
terms; it is independent of this adapter, and this adapter's gaps did not touch it.

## The Gate had never been run, and did not work

This adapter shipped with 14 negatives and a parity test, all green — and its primary
path had never been executed. The first time the CPU Gate was actually run inside a
container it failed twice:

1. `FileNotFoundError: docker`. `assert_not_host` permits only the inside-container
   mode, but `container_exec` shelled out to `docker exec`, which a container does not
   have. The adapter forbade the mode it supported and supported the mode it forbade.
   Fixed with a dual-mode path keyed on `/.dockerenv`.
2. `Failed to load the backend extension: torch_npu` — `gate_cpu` did not set
   `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, and torch autoloads the `torch_npu` backend. A CPU
   Gate must not load a device backend at all.

Every negative passed because they test configuration parsing and guard refusal. **None
of them executed the Gate against a real container**, which is exactly the path that was
broken. This is the same failure as the vacuous checks above, one level up: the suite was
green about everything except the thing the code exists to do.

The rule: **run the primary path once, end to end, before believing a test suite.** A
suite that never exercises it certifies only the side paths.

Verified after the fix, by execution inside `tf-ascend950pr`:

```
failures: 0
cpuEqualityRuns=True  checkCopySliceRuns=True  regressionsRun=True  verifyTirRuns=True
analyzeOk=True  finalizeOk=True  importOk=True  exitCodeZero=True
ranInsideContainer=True  inContainerPythonUsed=True  noHostInterpreter=True
```

## Every test here has been made to fail

A test that cannot fail certifies nothing, and it is worse than no test because it
produces confidence. Three checks in this adapter were written, passed, and were then
found to be incapable of failing:

| check | why it could not fail | what replaced it |
|---|---|---|
| busy-card refusal (grep) | a grep passed on both the refusal and the opt-in form because every string survived | `tests/device_policy.py`, reading the AST and asserting no busy branch, no wait, and the same workload for busy/idle |
| manifest/script parity (first version) | read only inline dicts, so it found zero keys and compared two empty sets | name resolution to the nearest assignment *before* each call |
| manifest/script parity (second version) | resolved to the last assignment in the file, so every Gate reported the final Gate's keys | position-scoped resolution |

The rule that follows: **prove the test fails before trusting it to pass.** Each of the
above was confirmed by deliberately breaking the thing it watches and observing the
failure, then restoring it. Those negative controls are part of the record:

```
busy -> skip mutation     -> FAIL "busy changed execution"
sleep/idle poll inserted  -> FAIL "a wait primitive appeared"
exclusive refusal restored -> FAIL "an exclusive dispatch branch appeared"
one declared key deleted  -> FAIL "declared but never reported: importOk"
restored                  -> 0
```

This matters most for properties a runtime assertion cannot observe. "Busy and idle
execute the same workload the same number of times" and "no branch can be conditioned
on occupancy" are structural claims. Grep is not enough: the regression they guard
against is a change of *structure* that preserves every string. Hence the AST plus an
executed two-state harness.

## Provision

```bash
scripts/setup-tf-env.sh --all      # or --cpu / --npu
```

Idempotent: it verifies what exists and installs only what is missing. If the
matching wheels are absent it exits **75 BLOCKED_ENV** and prints what it looked
for rather than substituting something else.

## Gates

```bash
python3 scripts/tilefoundry_gate.py --check-config   # configuration only
python3 scripts/tilefoundry_gate.py --build          # image + container
python3 scripts/tilefoundry_gate.py --cpu            # CPU Gate
python3 scripts/tilefoundry_gate.py --device --device-ids 0,1
```

Exit codes: `0` PASS, `1` FAIL, `2` configuration, `75` BLOCKED_ENV, `97` the
in-payload device assertion failed.

`--build` **reuses** `TF_REPO/docker/ascend950pr`. The adapter contains no
Dockerfile: the container that runs the Gate is the container the repo
verifies, which is the only version of that claim worth making.

## Configuration

`env.sh`, created from `env.example.sh`, `chmod 600`, git-ignored. It is parsed
as **data**, never sourced: unknown keys, duplicate keys and malformed lines all
fail. Exporting these names in a shell has no effect, so a stale export cannot
override what the file says the machine is.

`DEVICE_ID` is deliberately empty. A device list authorizes one run, not a
machine. Precedence is explicit: `--device-id/--device-ids` (an ordered comma
list) wins over `DEVICE_ID`, then the inherited `ASCEND_RT_VISIBLE_DEVICES`,
then the default `0,1`. Every requested id must be bound into the container; a
missing node is a real binding fault, not a busy condition.

## Device Gate

Device work enters the container as **root**. That is a platform fact, not a
convenience: the Ascend driver holds davinci cards per user namespace
(`uda_occupy_dev_by_ns`), so once any namespace holds the card a non-root
opener gets `EBUSY` even when it is in the device group. The Gate records that
it ran as root instead of hiding it.

The workload is real, not an import. For **every** device in the requested
list, in one payload, the Gate calls `torch.npu.set_device(logical)`, adds two
`float32` tensors of length 1024 on that device, synchronizes, reads the sum
back to the host, and checks it equals 3072.0. The per-device result carries
`logical`, `physical`, `device_name`, `count`, `sum` and `ok`. A device that
is requested but does not appear in the result makes `multiDeviceCoverage`
false, which fails the Gate; there is no first-device-only shortcut and no
import-only fallback.

The assertion runs **inside the container**, wrapping the payload, because the
wrapper's own environment says nothing about what the workload sees -- the
wrapper never enters the container. It sources `CANN_SET_ENV` first (without it
`torch_npu` cannot load `libhccl.so`, and `npu-smi` cannot load `libc_sec.so`),
asserts `ASCEND_RT_VISIBLE_DEVICES` equals the requested list **exactly**, in
order, and asserts uid 0. Any mismatch exits `97` before the workload starts.

Occupancy is **reported, never enforced**. This is a shared server; a busy card
is not a fact about the code under test. Busy and idle runs execute the same
workload, with the same verdict rule, and record `occupied`/holders as
non-verdict metadata. `TF_REQUIRE_EXCLUSIVE` and `TF_ALLOW_BUSY_DEVICE` are
retired legacy keys and are ignored.

One measured fact makes the metadata honest: `npu-smi` **inside this container**
reported `No running processes` on device 0 while the **host** reported four
processes on that same card, because the container sees only its own PID
namespace. So a holder row proves occupancy, and an empty table proves nothing:

```
occupancy per device: {"occupied": true|NONE, "holders": n|NONE,
                       "idle_established": false,
                       "scope": "container-pid-namespace"}
```

`occupied: NONE` means UNKNOWN, never "idle". The Gate does not act on it
either way, and it never re-samples until the table looks the way it prefers.

## Device attempt policy (ENV-05)

The device phase probes each visible card **individually**, with a bound, and
fails over only on a **driver-no-response** result.

| Parameter | Default | Env override | Why this value (measured on this host) |
|---|---|---|---|
| --device-attempt-timeout-seconds | **20 s** | TFNU_DEVICE_ATTEMPT_TIMEOUT | healthy npu-smi: host 0.77/0.78 s, container 1.02/1.00 s; a full device phase saw 1.4 s; worst HEALTHY observation 6 s. A D-state probe (wchan devdrv_sync_non_tran) took >20 s and could not be reclaimed after SIGKILL -- it is unbounded. 20 s separates the two. |
| --device-backoff-seconds | **30 s** | TFNU_DEVICE_BACKOFF | one natural window for transient congestion to clear before moving to the next card, without burning the run. |
| attempts | visible-list length | -- | today 2 cards, so worst case 2 x (20 + 30) ~= **100 s**, fully bounded. |

Rules (H addendum: attribution is strict):

| Probe result | Classification | Backoff / next card? | Run verdict |
|---|---|---|---|
| timeout **and** unreaped **and** a driver wchan (e.g. devdrv_sync_non_tran) | BUSY_DEVICE_NOT_RESPONDING | yes: one 30 s backoff, then the next card | NOT_RUN only if **all** cards are BUSY |
| timeout and unreaped, wchan missing/unsupported (cannot attribute to driver) | DEVICE_PROBE_UNATTRIBUTED | **no** | BLOCKED, named |
| timed out but child reclaimed (not unreaped) | DEVICE_PROBE_TIMEOUT | no | BLOCKED, named |
| fast rc != 0 (missing library/permission/command) | DEVICE_PROBE_FAILED | **no**; rc and stderr recorded | CONFIG/BLOCKED, named |
| rc == 0 but empty output | DEVICE_PROBE_EMPTY_OUTPUT | **no** | BLOCKED, named |
| responsive (even if occupied by another tenant) | RESPONSIVE | use immediately; no retry/refusal | normal phase continues |

- A card that is merely **occupied** by another tenant is normal shared use: it is
  recorded, never a retry/refusal/switch reason.
- Only BUSY_DEVICE_NOT_RESPONDING may fail over. Any other non-responsive result
  stops the run at that card (attempt count = 1, no backoff, no next card).
- Every card is tried **once** per run; the attempt count is bounded by the
  visible list.
- After a failover, the selected card performs its **own fresh**
  launch -> torch.npu.synchronize() -> host per-element readback. A previous
  card's result is never reused.
- Every attempt is written to device_attempts with
  logical_index/physical_id/attempt/timed_out/unreaped/wchan/pid/elapsed, plus
  selected_device and all_cards_unresponsive.
- If **every** card is unresponsive: NOT_RUN + BUSY_DEVICE_NOT_RESPONDING +
  all_cards_unresponsive=true. That is a declared failure of the environment,
  never a PASS and never a software FAIL.

TFNU_PROBE_OVERRIDE (e.g. 0=unresponsive,1=responsive) is a TEST seam for the
two failover negatives; every simulated attempt carries simulated=true.

## H device harness (adapter side, harness-run/1)

The H harness is a thin layer over the canonical device entry; it never opens a second
ENV path and never substitutes a generic add for an unregistered workload.

- **Workloads** are registered by ID with a canonical recipe + X mode + output spec;
  an unknown ID is `UNSUPPORTED_WORKLOAD`, and any other/generic entry is
  `NON_CANONICAL_DEVICE_ENTRY`.
- **Scope is separate from selection**: `SELECT_ONE` reports only the selected card;
  `ALL_REQUESTED` requires a real, distinct output for every requested physical id,
  otherwise `DEVICE_COVERAGE_MISMATCH` (another card never substitutes).
- **Failover is pre-launch only**: only a driver probe that is `timed_out + unreaped +
  driver wchan` may move to another card. Once launch starts, a timeout/error/numeric
  failure is `LAUNCH_FAILED`/`SYNC_FAILED`/`NUMERIC_MISMATCH` in the original
  scope; it is **not** retried elsewhere.
- Phases, raw actual inputs + eps f32 bits, raw outputs, and the four artifact
  identities `source/CCE/binary/load` (OBSERVED or UNOBSERVED) are written under the
  run directory; every case gets its own directory.
- **FINISHED is not PASS**: `finished` means the run ended, including failures;
  `passed` additionally requires all required phases, complete coverage, observed
  artifact identities, raw output bytes, and a passing T report. Exit 0, a finished
  flag, an all-skipped run, or a stale payload cannot pass.
- Local evidence and any AgentGates service execution/receipt are separate fields; an
  unregistered local run reports `service_receipt=None` and never invents a run id.
- CPU verify stays `--skip-device` by default; a device recipe needs an explicit
  `allow_device=True` request.

## What a PASS does not mean

- **It is not device attestation.** The receipt is written by the same process
  it describes; a schema check cannot detect a deliberate one.
- **It is not a service receipt.** This adapter is not registered with the
  AgentGates service; local JSON is evidence, not a signed gate record.
- **A timing number is provisional** on a shared card and is never a claim
  about the hardware.
- **A PASS only proves this run, as recorded, under this configuration.**

## Negatives

```bash
tests/negatives.sh
```

Covers: missing config, unknown key, duplicate key, empty required value, a repo
without the docker recipe, host invocation refused, device Gate with an empty or
non-numeric list, a duplicate id, and no subcommand selected.