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

### A held card is refused, not entered

Before anything imports `torch_npu` or reaches `aclInit`, the device Gate reads
occupancy. If another tenant already holds the card it **refuses with exit 75
`BLOCKED_ENV`** and the workload never runs.

This is the default, and it was learned the hard way. In this program a device open at
`13:46:26` was followed **32 s later** by a host-wide `docker exec` stall, with the
card already held by other tenants. A failed `aclInit` still *opens the device node*, so
"it only errored" is not a defence: the open happens either way.

The ordering is the point, so the check lives **above** the probe and must not be moved
below it. The previous default was the opposite — record occupancy, proceed, refuse only
when `TF_REQUIRE_EXCLUSIVE=1` — which made walking onto somebody else's card the normal
case.

An override exists and is deliberately loud:

```
TF_ALLOW_BUSY_DEVICE=1
```

It prints an unmissable banner, records `overriddenBusyDevice=true` in the evidence, and
the result is labelled a **shared-card** result: timings provisional, numerics
unaffected. The flag is named for what it does, so nobody sets it by accident. There is
no way to reach a device while hiding that the card was busy.

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
| busy-card refusal (grep) | reverting the refusal to the opt-in form keeps every string a grep looks for, so it passed on the regression it existed to catch | `tests/device_policy.py`, reading the AST and asserting ORDER and DEFAULT |
| manifest/script parity (first version) | read only inline dicts, so it found zero keys and compared two empty sets | name resolution to the nearest assignment *before* each call |
| manifest/script parity (second version) | resolved to the last assignment in the file, so every Gate reported the final Gate's keys | position-scoped resolution |

The rule that follows: **prove the test fails before trusting it to pass.** Each of the
above was confirmed by deliberately breaking the thing it watches and observing the
failure, then restoring it. Those negative controls are part of the record:

```
opt-in revert            -> FAIL "the refusal is conditional on an opt-in flag" (1)
occupancy moved later    -> FAIL on order (1)
one declared key deleted -> FAIL "declared but never reported: importOk" (1)
restored                 -> 0
```

This matters most for properties a runtime assertion cannot observe. "A held card
returns 75" is testable at runtime; "occupancy is read **before** the device is opened"
and "the refusal is not conditional on an opt-in flag" are not, because both are about
code that has not run yet. Grep is not enough for those either — the regression they
guard against is a change of *structure* that preserves every string. Hence the AST.

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
python3 scripts/tilefoundry_gate.py --device --device-id 0
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

`DEVICE_ID` is deliberately empty. A device id authorizes one run, not a
machine, so it comes from `--device-id` or `DEVICE_ID` and is never discovered
or defaulted.

## Device Gate

Device work enters the container as **root**. That is a platform fact, not a
convenience: the Ascend driver holds davinci cards per user namespace
(`uda_occupy_dev_by_ns`), so once any namespace holds the card a non-root
opener gets `EBUSY` even when it is in the device group. The Gate records that
it ran as root instead of hiding it.

The assertion runs **inside the container**, wrapping the payload, because the
wrapper's own environment says nothing about what the workload sees -- the
wrapper never enters the container. It reads `ASCEND_RT_VISIBLE_DEVICES` as the
workload will see it, requires exactly one device matching the authorized id,
requires uid 0, and exits `97` before the workload starts on any mismatch.

Occupancy is **reported, not enforced**. This is a shared server; a busy card is
not a fact about the code under test. Timings taken while another tenant is
active are therefore labelled provisional, and numerics are unaffected by
sharing. `TF_REQUIRE_EXCLUSIVE=1` turns a held card into exit **75** -- not a
pass, not a failure: there is no result to report.

## What a PASS does not mean

- **It is not device attestation.** The receipt is written by the same process
  it describes; a schema check cannot detect a deliberate one.
- **A timing number is provisional** on a shared card and is never a claim
  about the hardware.
- **A PASS only proves this run, as recorded, under this configuration.**

## Negatives

```bash
tests/negatives.sh
```

Covers: missing config, unknown key, duplicate key, empty required value, a repo
without the docker recipe, host invocation refused, device Gate without an id,
a non-numeric id, and no subcommand selected.