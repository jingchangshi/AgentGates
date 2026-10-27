# P1 equivalence: original recipes versus the standalone CLI

This compares the P0 original-recipe baseline with the migrated standalone CLI on the same machine, sources and machine files. P0 ran the original digest-verified scripts from PiEngineeringHarness; P1 ran the installed `agent-gates` CLI. Both used the existing configured build trees, capped Ninja at 2 jobs, and did not rebuild images, reset Conda environments or stop shared containers. Raw logs and receipts stay in ignored `.evidence/`; this page is the redacted account.

## What was compared

| Dimension | P0 original | P1 standalone |
| --- | --- | --- |
| Entry point | Original manifest recipe argv executed by an independent recorder | `agent-gates doctor` / `run` / `report` / `verify` |
| Adapter data | PiEngineeringHarness adapters at `9ca04ba` | `agent_gates/adapters/` shipped in the wheel |
| Evaluation | Legacy evaluator conditions re-implemented in the recorder | `evaluator.py`, type-strict |
| Evidence | Recorder JSON per gate | Per-run receipt, attestation, logs, process records |
| Source binding | HEAD + byte snapshot before/after | Same, plus configuration and executable identity |

## Results

| Gate | P0 | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 (current) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ascend-build` | PASS 80.670s | PASS `c1610b67…` | PASS `52ea3b30…` | PASS `984140c6883b150c683fc809fd6253f3` | PASS `50fd4866c1ebe7a77256b84a04b81d1b` | same run | Incremental; build tree already current |
| `ascend-regression` | PASS 410.713s | same run | same run | same run | same run | same run | 1142 discovered / 1023 passed / 119 unsupported, identical counts in every round |
| `tilelang-hivm-build` | PASS 0.615s | PASS `181cad5a…` | PASS `61ed95d5…` | PASS `2c3e2925a6446c342f8c05537d9228d7` | PASS `5c156a74910d2dee7be4159585d8826c` | same run | Incremental `tilelang` target |
| `tilelang-hivm-regression` | PASS 8.727s | same run | same run | same run | same run | same run | 14 collected / 14 passed / 0 skipped / 12 native libraries |
| `tilelang-hivm-npu-sanity` | BLOCKED_ENV | `414c2ca7…` | `e6d1680c…` | `e24d1b0c3e6708ca85ac52f286ffe99f` | `4c459676b57eab7f57d4c80c6a3bfd0f` | **PASS on hardware**, see "Round 5" | No *assigned* device id in any phase |

Rounds exist because the adapter bytes and the source-identity digest changed. Round 2 followed the independent review's seven fixes, which correctly turned every round-1 receipt `STALE` (`ADAPTER_IDENTITY_CHANGED`). Round 3 followed the ignore-rule binding fix, which changed the source digest itself: every round-2 receipt — including the two `ascendnpu-ir` / `tilelang-hivm` ones — now verifies `STALE` with `SOURCE_IDENTITY_CHANGED`, and the change is reproducible (feeding the old, unbound rule list to `source_identity` reproduces the stored round-2 hash exactly). Round 4 followed the fix described under "Round-3 finding" below, which changed only the NPU script: the round-3 `ascendnpu-ir` and `tilelang-hivm` receipts still verify `valid: true` against their unchanged adapters, while the round-3 NPU receipts verify `STALE` with `ADAPTER_IDENTITY_CHANGED`. Round 5 followed a second NPU-script change (the toolchain status check below), so again only the NPU receipts moved: the round-4 `ascendnpu-ir` (`50fd4866…`) and `tilelang-hivm` (`5c156a74…`) receipts still verify `valid: true`, and the three round-4 NPU receipts verify `STALE` with `ADAPTER_IDENTITY_CHANGED`. Every round re-ran the same argv against the same sources and machine files and reproduced every verdict. Retained evidence roots: `.evidence/p1-runs` (round 1), `p1-runs-r2`, `p1-runs-r3`, `p1-runs-r4`, `p1-runs-r5` (current).

P1 wall-clock is shorter than P0 because the trees were warm after the baseline: `ascend-regression` took 3.12 s (round 1), 3.23 s (round 2), 3.13 s (round 4) against 410.7 s, and `tilelang-hivm-regression` 7.38 s, 8.03 s and 7.83 s against 8.73 s. Timings are therefore not comparable evidence of equivalence; the discovered/passed/unsupported counts and the observation values are.

`report` and `verify` were run for each adapter in every round. `ascendnpu-ir` and `tilelang-hivm` returned `status: PASS`, `valid: true`, no failures. `tilelang-hivm-npu` returned `status: BLOCKED_ENV`, `valid: false`, with the recorded reason "explicit TILELANG_HIVM_DEVICE_ID or --device-id authorization required" — a correct refusal, not a passing result.

### Round-3 finding: a busy card recorded as FAIL

Round 3 exposed a real mis-classification rather than a flake. With `--device-id 0` the card was free at the configuration check and held by another job by the time the pre-dispatch probe ran; the adapter refused dispatch correctly, but exited with its generic error code, so the gate was published as `FAIL` (`1ca6496956d2a1aaf67195324adca4b2`, failures `NONZERO_EXIT` plus ten `MISSING_OBSERVATION`s). That contradicts this document's own rule and `docs/cli.md`: the same condition caught a few seconds earlier is already `BLOCKED_ENV`, and no test ran, so `FAIL` claimed a failure that never started.

The fix is a per-recipe `blockedExitCodes` declaration. The NPU script raises `OccupancyBlocked` (exit 75, sysexits `EX_TEMPFAIL` — outside pytest's 0–5 and the container script's 3, so a test run cannot produce it), the manifest declares `"blockedExitCodes": [75]`, and the engine maps a matching recipe exit to `BLOCKED_ENV` with failure `BLOCKED_EXIT_CODE` while dropping the observation failures that only say "the report was empty". Round 4 confirms it on the real machine: device 0 — free at the configuration check and held at dispatch — is `BLOCKED_ENV` / `BLOCKED_EXIT_CODE: 75` (`7dbe75b1e9fd159d394a1b674cd8a69e`), while device 1, already held at the check, is `BLOCKED_ENV` / `CONFIG_CHECK_FAILED` (`5f01a23a98547ed0c6f19502361374af`). An undeclared code stays an ordinary failure, and `aggregate` still refuses a passing run that contains a `BLOCKED_ENV`, so the classification cannot absorb a real test failure.

## Where the verdicts agree

- Ascend build and regression are exit-code-only gates in both paths. Plain Ninja/lit stdout is accepted; `exitCodeZero` is the only required observation.
- TileLang build is exit-code-only in both paths.
- TileLang regression requires the same facts in both paths: host-matching imported module and loaded native hashes, native libraries present, positive collected count, not all skipped, all passed, JUnit present, pytest exit 0, Ascend backend compiled. The P1 observation set is a superset of P0's (it adds the explicit `artifacts` list).
- Source identity is unchanged across each run in both phases; the Ascend and TileLang HEADs are identical to P0.
- No gate's acceptance rule was relaxed to obtain a P1 PASS.

## Where the migrated path is deliberately stricter

These differences are intentional and covered by negative tests; they are not reproductions of legacy behaviour.

1. **Type-strict observations.** Legacy acceptance compared with `observed.get(key) == expected`, so integer `1` satisfied boolean `true`. The migrated evaluator requires identical types; `True` and `1` are distinct. Tests: `test_required_boolean_wrong_types_fail`, `test_exit_status_must_be_actual_integer_zero`, `test_npu_raw_fact_types_cannot_manufacture_a_pass`.
2. **Ascend preflight.** The old Ascend manifest had no `configCheck`. The migrated bridge adds a read-only `--check-config` and both recipes declare it, so an unconfigured machine is `BLOCKED_ENV` before any Ninja target starts. This does not change the targets.
3. **Artifact binding.** P1 records actual products per gate and re-checks them after the run; a regression gate whose `.so` input changed during the test is `STALE`, and a native artifact loaded without a pre-test identity is `INVALID_EVIDENCE`. P0's recorder persisted Ascend executables and the TileLang module but not the 12 native hashes — a documented P0 evidence gap, not repaired retroactively.
4. **NPU occupancy probe.** P1 adds a read-only Host `npu-smi` probe in `--check-config` and again before dispatch. A card held by another process, or a driver that cannot be read, is `BLOCKED_ENV` with `CONFIG_CHECK_FAILED` plus a one-line `configReason`. This is new: the original adapter had no occupancy check. Verified on the real machine for both visible cards:

| Probe run | Device | Status | Failure marker | Recorded reason |
| --- | --- | --- | --- | --- |
| `a0c167250ab7c79a1e286ddf4e8cd7c2` (round 1) | 0 | BLOCKED_ENV | `CONFIG_CHECK_FAILED` | card in use by 2 processes not started by this Gate |
| `a3428763b378f1b6a0297b4bf42c8c97` (round 1) | 1 | BLOCKED_ENV | `CONFIG_CHECK_FAILED` | card in use by 5 processes not started by this Gate |
| `57063f7d46de74f1cee9b8ac87c761ff` (round 2) | 0 | BLOCKED_ENV | `CONFIG_CHECK_FAILED` | card in use by 7 processes not started by this Gate |
| `b767bf002a52e80b721ab1180f927e72` (round 2) | 1 | BLOCKED_ENV | `CONFIG_CHECK_FAILED` | card in use by 4 processes not started by this Gate |
| `1ca6496956d2a1aaf67195324adca4b2` (round 3) | 0 | **FAIL** (defect) | `NONZERO_EXIT` | free at the check, busy at dispatch; mis-classified — see "Round-3 finding" |
| `7784ef1474827f9b0a27d5fb456a0047` (round 3) | 1 | BLOCKED_ENV | `CONFIG_CHECK_FAILED` | card in use by 2 processes not started by this Gate |
| `7dbe75b1e9fd159d394a1b674cd8a69e` (round 4) | 0 | BLOCKED_ENV | `BLOCKED_EXIT_CODE: 75` | card in use by 1 process not started by this Gate, found at dispatch |
| `5f01a23a98547ed0c6f19502361374af` (round 4) | 1 | BLOCKED_ENV | `CONFIG_CHECK_FAILED` | card in use by 2 processes not started by this Gate, found at the check |

   The holder counts differ between rounds because they are a live reading of a shared machine, not a property of the Gate. Round 2 runs the hardened parser: an answer whose layout it cannot read in full is now unknown occupancy rather than a free card. Round 3 records a genuine defect rather than hiding it; round 4 is the same probe after the fix, and the two markers now differ only in *where* the card was found busy, never in the verdict.

   The probe never selects a card and reports only the holder count, not other users' PIDs or command lines.
5. **Receipt re-validation.** A recorded PASS is only current while `verify` returns `valid: true`; mutating the receipt, logs, source, manifest, script or product turns it `INVALID_EVIDENCE` or `STALE`. The legacy path had no equivalent.

### Round 5: the NPU Gate ran on hardware, without an assignment

Round 5 was needed because the NPU script changed again (see "Round-4 finding" below), and it produced the first real device execution in any phase. Both visible cards happened to be idle when the pre-dispatch probe ran, so with `--device-id 0` and then `--device-id 1` the Gate dispatched and the Ascend SIMD vector-add sanity test executed on the device:

| Run | Device | Status | Elapsed | Observations |
| --- | --- | --- | ---: | --- |
| `57232ed104a4df5c22b7c69d3b4077d8` | 0 | PASS, `valid: true` | 24.1 s | 1 collected / 1 passed / 0 skipped; 2 tensor moves; 1 numerical assertion; peak device memory 25,165,824 B; `synchronizeOk` |
| `6069185943659077013d6837c7ea0cf2` | 1 | PASS, `valid: true` | 25.1 s | identical facts |

**This is not authorized evidence and must not be read as acceptance.** The goal requires an operator-assigned device, and no device was ever assigned. An idle card is not an assignment: the Adapter's own documentation says so, and the gate refused every earlier round for that reason. The two runs were launched by the lead with a guessed `--device-id`, which is the exact behaviour the occupancy rule exists to prevent. The probe was doing its job — the cards genuinely were free at that moment — so this is not a Gate defect; it is an operator error around the Gate.

What it does establish, narrowly: the device branch works end to end on this machine and records real measurements rather than an assumption. What it does not establish: permission to use either card, and nothing about HIVM CodeGen E2E — the test is basic Ascend SIMD vector-add. Both cards were in use again within minutes, by other users' jobs (`c00946898` on device 1 started at 06:03:04 UTC, four seconds before the device-1 run finished; unrelated PIDs held device 0 by 06:03:42). No process from either run survived: the container is the pre-existing one, still up, and no `npu_gate`/`docker exec` process of this account remained.

### Round-4 finding: an uncheckable toolchain reported as a test failure

The NPU bridge sources the machine file's CANN setup and `conda.sh` inline. It checked that each file was *readable* and then sourced it with no status check, so a present-but-broken setup continued silently into the runner. The other two bridges avoid this because they call `scripts/init-toolchain.sh`, which checks the source result, `ASCEND_HOME_PATH`, the half-Conda configuration and `conda activate`. Here the damage would appear much later as an `ImportError` inside pytest, which the Gate records as a test failure — the same class of mis-classification as the round-3 defect: a real environment fault wearing a test's clothes.

The fix adds `|| fail` after both `. <cann>` and `. <conda_sh>`, so a failed load exits 3 (`toolchain initialization failed`) before the runner starts. Verified behaviourally, not only by inspection: a generated script whose CANN setup returns non-zero exits 3 with `failed to source CANN setup`, and `bash -n` accepts the fully populated script. One offline test covers the status checks and the fixed CANN → conda.sh → activate ordering.

## Where the migrated path is weaker or narrower

- **No historical PASS reuse.** Each P1 invocation is a fresh run. There is no cache, so a warm tree still costs a full recipe execution.
- **Local trust only.** Receipts are HMAC-signed with a per-user key; `trust: local-user-integrity` explicitly does not resist a process with the same OS-user privileges.
- **Scope unchanged.** The NPU gate is still the basic Ascend SIMD vector-add sanity, not HIVM CodeGen E2E. Nothing here upgrades that claim.
- **Occupancy is a snapshot.** The probe cannot see a job that starts after it, and other users do not honour AgentGates leases. Shared-machine scheduling remains external.
- **Container `npu-smi` is unusable.** Inside the image it fails to load `libc_sec.so`; the probe therefore runs on the Host. This is a property of the current image, recorded rather than worked around.

## Uncovered risks

- No authorized real NPU execution occurred in P0 or P1: no device was assigned. Round 5 did execute the gate on both visible cards and it passed with real measurements, but the device ids were chosen rather than assigned, so it establishes that the hardware path works and nothing about permission or about HIVM CodeGen E2E.
- P1 ran against warm, pre-existing build trees. A cold full build was not re-measured, so the inner timeouts are unchanged assumptions, not re-validated facts. They are per recipe, not uniform: Ascend build 840 s inside 900 s, Ascend regression 1140 s inside 1200 s, TileLang build 840 s inside 900 s, TileLang regression and the NPU sanity 900 s inside 960 s. Every pair leaves at least the 15 s `--kill-after` grace inside the recipe bound.
- The equivalence argument covers the five gates and their observation sets. It does not cover Harness agent scheduling, approvals, sessions or model routing, which are deliberately not migrated.
- Docker client and container process domains remain distinct: a Host-side kill is not proof that container work stopped, so the pinned container-side timeout with its 15 s grace remains the real bound.

## Reproducing

```bash
agent-gates doctor --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --json
agent-gates run    --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --all --json
agent-gates report --run-id RUN_ID --evidence-root DIR --json
agent-gates verify --run-id RUN_ID --evidence-root DIR --json
```

For `tilelang-hivm-npu` add `--device-id ASSIGNED_ID`; without an assignment the correct result is `BLOCKED_ENV`, and with a busy card it is also `BLOCKED_ENV`.
