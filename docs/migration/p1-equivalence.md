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

| Gate | P0 | Round 1 | Round 2 | Round 3 | Round 4 (current) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ascend-build` | PASS 80.670s | PASS `c1610b67…` | PASS `52ea3b30…` | PASS `984140c6883b150c683fc809fd6253f3` | PASS `50fd4866c1ebe7a77256b84a04b81d1b` | Incremental; build tree already current |
| `ascend-regression` | PASS 410.713s | same run | same run | same run | same run | 1142 discovered / 1023 passed / 119 unsupported, identical counts in every round |
| `tilelang-hivm-build` | PASS 0.615s | PASS `181cad5a…` | PASS `61ed95d5…` | PASS `2c3e2925a6446c342f8c05537d9228d7` | PASS `5c156a74910d2dee7be4159585d8826c` | Incremental `tilelang` target |
| `tilelang-hivm-regression` | PASS 8.727s | same run | same run | same run | same run | 14 collected / 14 passed / 0 skipped / 12 native libraries |
| `tilelang-hivm-npu-sanity` | BLOCKED_ENV | `414c2ca7…` | `e6d1680c…` | `e24d1b0c3e6708ca85ac52f286ffe99f` | `4c459676b57eab7f57d4c80c6a3bfd0f` | No task-assigned device id in any phase |

Rounds exist because the adapter bytes and the source-identity digest changed. Round 2 followed the independent review's seven fixes, which correctly turned every round-1 receipt `STALE` (`ADAPTER_IDENTITY_CHANGED`). Round 3 followed the ignore-rule binding fix, which changed the source digest itself: every round-2 receipt — including the two `ascendnpu-ir` / `tilelang-hivm` ones — now verifies `STALE` with `SOURCE_IDENTITY_CHANGED`, and the change is reproducible (feeding the old, unbound rule list to `source_identity` reproduces the stored round-2 hash exactly). Round 4 followed the fix described under "Round-3 finding" below, which changed only the NPU script: the round-3 `ascendnpu-ir` and `tilelang-hivm` receipts still verify `valid: true` against their unchanged adapters, while the round-3 NPU receipts verify `STALE` with `ADAPTER_IDENTITY_CHANGED`. Every round re-ran the same argv against the same sources and machine files and reproduced every verdict. Retained evidence roots: `.evidence/p1-runs` (round 1), `p1-runs-r2`, `p1-runs-r3`, `p1-runs-r4` (current).

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

## Where the migrated path is weaker or narrower

- **No historical PASS reuse.** Each P1 invocation is a fresh run. There is no cache, so a warm tree still costs a full recipe execution.
- **Local trust only.** Receipts are HMAC-signed with a per-user key; `trust: local-user-integrity` explicitly does not resist a process with the same OS-user privileges.
- **Scope unchanged.** The NPU gate is still the basic Ascend SIMD vector-add sanity, not HIVM CodeGen E2E. Nothing here upgrades that claim.
- **Occupancy is a snapshot.** The probe cannot see a job that starts after it, and other users do not honour AgentGates leases. Shared-machine scheduling remains external.
- **Container `npu-smi` is unusable.** Inside the image it fails to load `libc_sec.so`; the probe therefore runs on the Host. This is a property of the current image, recorded rather than worked around.

## Uncovered risks

- No real NPU execution occurred in P0 or P1: both cards are held by other users' jobs and no device was assigned. The NPU gate's hardware behaviour is covered only by offline fixtures and negative tests.
- P1 ran against warm, pre-existing build trees. A cold full build was not re-measured, so cold-build timeouts (840 s / 1140 s inner) are unchanged assumptions, not re-validated facts.
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
