# tla-npu adapter: plan for Agent-driven TLA operator generation and performance optimisation

Status: proposal, reviewed. Nothing in this document is implemented yet.

Decisions taken by the operator (2026-11-02):

- **P1 compromise accepted**: the upstream suite gates at stage `ir` only; stage `run` is
  recorded as a baseline with an explicit `expectedFailures` set, never gated.
- **P3 is in scope**: the harness core (`api_types.py`, `mcp_server.py`) will be changed so
  observations reach an Agent through MCP.
- Implementation does not start until the architecture review below is answered.

## 1. Where the adapter stands (verified on this machine)

| Fact | Value |
|---|---|
| Source root under test | `/home/s00653124/workspace/triton-ascend`, `main-dev`, head `2a00c605`, 3275 files, `git-files` mode |
| Test directory (separate input) | `TLA_NPU_TEST_DIR` -> `/home/s00653124/workspace/agent_tla` |
| Cache (separate, refused inside either input) | `TLA_NPU_CACHE_DIR` -> `/home/s00653124/.cache/tla-npu-gate` |
| Triton wheel | `dev20261101+git2a00c605`, pinned by path in `env.sh` |
| TLA implementation identity | recorded: `tlaImplementationSha256`, 21 files, `tritonVersion 3.6.0`, `tlaHasDma True` |
| Current gates | one: `tla-npu-e2e` (deterministic-test), 11 boolean checks |
| Last run | PASS, `verify` -> `valid: true` |
| Harness suite | 655 passed / 88 subtests |

What it proves today: one fixed script, `test_tla.py`, ran on one named card and produced
the expected numerical markers. That is a device regression gate.

## 2. What the intended workflow actually needs

The goal is an Agent that writes TLA kernels and makes them faster. The loop is:

1. Agent writes or edits a TLA kernel.
2. Correctness is checked. A kernel that computes the wrong answer is not an optimisation.
3. Performance is measured against a baseline.
4. The agent decides whether to keep the change and iterates.

Step 1 is where the current adapter does not fit: it runs **one fixed file**. An agent
optimising its own operator needs the adapter to exercise *the agent's* kernels, and to
report per-kernel results rather than a single pass/fail.

## 3. The hard constraint: this plan refuses a performance-threshold gate

This shapes everything below, so it comes first.

`agent_gates/evaluator.py` compares each declared observation with **exact equality**
(`equal()`: type identity, then `==`). It has exactly two special cases, both for test
counts (`testsCollectedPositive`, `notAllSkipped`). There is no support for a range, a
minimum, a tolerance, or a comparison against a previous run. `manifest.py` also refuses
any `gateType` other than `deterministic-test`.

Correction from review, and it matters for how this is argued: `manifest.py` does **not**
type-check the values in `observationChecks`, so `{"speedupAtLeast": 1.2}` **loads fine**
(verified). It then fails forever, because `equal()` demands type identity and a float is
never a `true`. So a threshold is not *impossible* - it is *silently unexpressible today*,
which happens to be fail-closed. One operator in `evaluate()` would make it work.

**This plan refuses that operator as a deliberate decision, not because the code forbids
it.** Putting a sample-statistic arbitration rule inside `evaluate()` would convert a
median into a deterministic verdict on a card whose cross-process drift is 25.8%.

Consequently a gate in this plan **will not** assert "speedup >= 1.2". Writing `speedupAtLeast1_2: true`
would only assert that the adapter printed `true`, which is the tautology already removed
from this adapter once (the old `deviceMatchesAuthorized`).

**Therefore the division is:**

- **Correctness gates** assert exact booleans and can fail the run.
- **Performance measurements** are *recorded* in the receipt, exactly as
  `tlaImplementationSha256` is recorded today, and never gate.
- The **speed verdict** is made by a comparison step against a baseline, outside the gate,
  by a tool that reads two receipts.

Recording without gating is not a downgrade. The receipt is the auditable artifact; a
measured number bound to a specific source commit, wheel and device state is exactly what
a human or an Agent needs to judge a change. Pretending a threshold gate is possible would
be the defect.

## 4. The measurement constraint: noise decides the method

Measured on this machine, NPU 1, `triton.testing.do_bench` (it works on Ascend: it goes
through `runtime.driver.active.get_device_interface()`, which exposes `Event` and
`synchronize`):

| Measurement | Observed spread |
|---|---|
| Same process, repeated `do_bench` | 0.21% |
| **Separate processes**, same kernel | **25.8%** (0.012840 - 0.016155 ms) |
| Same process, interleaved A/B, ratio | 0.18 - 0.23% |
| Separate processes, ratio medians | 1.1758 / 1.1384 / 1.1452 -> ~3.3% drift |

Two conclusions:

1. A number measured in one process **cannot** be compared with a number measured in
   another. Wall-clock baselines stored in a file are worthless at this granularity.
2. Baseline and candidate **must be measured in the same process, interleaved**, and the
   reported quantity must be their **ratio**, not an absolute time.

Third factor: this card is shared. `npu-smi` shows other tenants routinely, and a shared
card adds jitter that no methodology can remove. So a performance measurement must run
with **exclusive device access**, which correctness does not require. The adapter already
has the mechanism: `TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE`.

Device context is readable host-side and should be recorded with every measurement
(`npu-smi info -t temp/-t power/-t usages`): at time of writing NPU 1 reported 63 C,
205 W, Aicore 0%, HBM 4%. Temperature and power are the covariates that explain a slow
run.

## 5. Plan

### P0 - BLOCKED until the test bytes are in receipt identity

The review found the most serious defect in this plan here, and it is confirmed against
the live receipt (`d0f6d5c5df0c0d476f471958f15c54de`):

- `source.root` is the triton-ascend checkout; `source` has keys root/head/hash/mode/
  fileCount/symlinks. **No test directory.**
- Searching the whole receipt for `agent_tla`: **0 matches**.
- Searching all evidence for the test's own sha256 `e0b4f475...`: **0 matches**.
- `engine.py` snapshots only `source`, before and after.

So the receipt binds the implementation it claims to test, and does **not** bind the bytes
that do the testing. Generalising one unhashed file into an Agent-controlled unhashed
directory turns a single gap into an unbounded one. P0 must not be implemented before that
hole is closed.

The rest of P0 keeps its shape, with the review's corrections:

- **No discovery.** An Agent that controls the directory could satisfy any count with one
  trivial case. Use a fixed, enumerated case list; the gate hashes each case file and
  records the hashes and a count.
- **Integer guards, not booleans.** `casesPassed == casesDiscovered == 0` would otherwise
  pass vacuously, and the evaluator's cross-checks only fire for the literal keys
  `testsCollectedPositive` / `notAllSkipped`, so this gate must carry its own integer
  guard: `casesDiscovered > 0`, `casesRun <= casesDiscovered`, `casesPassed == casesRun`.
- **No silent skip.** Record `casesSkipped` and fail on any case that neither ran nor was
  explicitly declared skipped.
- **Re-verify case bytes after the run**, the same way the source is re-verified, so a case
  edited mid-run is STALE.

### P1 - Correctness at scale: the vendor TLA suite (baseline corrected)

The review is right that a name-pinned `expectedFailures` list rots: pytest ids like
`simd_tail.py::test_run[f32-n64]` are parametrised, so one upstream `parametrize` addition
shifts every key and a **new** real failure can cancel against a removed one - net
unchanged, undetectable. It also could not confirm my "21" is stable (same case reported
"absolute 17.0 / relative inf" on one run, different pytest text on another; 20.98s vs 92s
wall). No number gets pinned before its stability is measured.

Corrected design: gate stage `ir` only (verified clean). For stage `run`, compare the
**hash of the failing-id set** against a committed baseline and assert "no NEW failure",
flagging baseline staleness on drift and reporting BLOCKED with a reason rather than green.
The 21 upstream failures are reported upstream, not normalised.

### P2 - Measure, and record the measurement (methodology corrected)

The review dismantled the measurement design. Every point below was verified against
`python/triton/testing.py` and this adapter:

| Claim in the first draft | Reality |
|---|---|
| `warmup`/`rep` are iteration counts | They are **milliseconds**; `n_repeat = rep / estimate_ms` |
| default statistic is fine | default `return_mode` is **mean**, the statistic a shared-tenant spike corrupts; must be `median` |
| interleaved A/B is enough | `do_bench` calls `fn()` once to estimate, so **only the first kernel is warm**; every kernel must be warmed first or kernel 2's compile lands in kernel 1's number |
| exclusive device gives isolation | **Not enforceable here.** `require_device_free` is a start-time check and the `npu:<id>` lease is a cooperative flock among AgentGates runs only. Another tenant can start mid-benchmark. |
| ~14us kernels illustrate optimisation | At that size you are likely timing launch/dispatch, not the kernel; ratios do not generalise |

Corrected design:

- Explicit `warmup`/`rep` in **milliseconds**, `return_mode="median"`, every kernel warmed
  before any timing.
- Record the cache state: a per-run `TRITON_CACHE_DIR` (`mktemp -d` or the configured
  `TLA_NPU_CACHE_DIR`), because two kernels starting from two cold caches is guaranteed
  bias.
- **Drop the exclusivity claim.** Instead record actual occupancy at start and end, and
  gate on measurement *usability* (`noiseExceedsEffect` false, all timings finite, spread
  reported) - never on the ratio's value.
- Record device context (`deviceTempC`, `devicePowerW`, `aicoreUsagePct`) as covariates
  **and** require comparability before any comparison is drawn (see P4).
- Report kernels large enough that the measured quantity is the kernel, or state plainly
  that small-kernel ratios measure dispatch.

### P3 - Let the Agent see the numbers

Gap found: `mcp_server.py`'s `GateResult` and `ReportView` expose gate id, status,
failures and recipe id - **not observations**. The numbers in P2 would be invisible to an
Agent driving through MCP, which is the intended consumer.

**Review correction: bounded is not typed.** `strict_json` dedups keys and rejects NaN/Inf,
but constrains neither depth, key count, nor string length. The adapter emits 21 keys today;
an Agent-authored case could print millions into a model's context, and
`tlaImplementationError` is already free text (a latent injection vector). So `GateResult`
must expose a **fixed named allowlist of scalars with per-field caps** - a projected set,
never the raw dict.

Add an observations field to `GateResult` (bounded, from the receipt as already stored via
`gate.update(evaluate(...))`). This is a small, additive change to `api_types.py` and
`_receipt_report`. It touches shared code, so it needs the same review as any core change,
and the field must stay a plain dict of scalars - no free-form blob.

Alternative if core changes are out of scope now: the Agent reads the receipt file
directly. Workable, but then the MCP surface answers "did it pass" and cannot answer
"how fast", which is half the workflow.

### P4 - The speed verdict, as a tool and not a gate

Ship `scripts/perf_compare.py` in the adapter pack: given two run ids (or two receipts),
emit a verdict with the ratio, its spread, and whether the difference exceeds noise, the
same rule P2 reports. It reads only published receipts, so it cannot fabricate evidence.

**Review correction: a tool that always prints a ratio is a tautology in a lab coat.** The
ratio is only meaningful if the devices are comparable, and recording `deviceTempC`/
`devicePowerW` without a matching requirement repeats the `deviceMatchesAuthorized` defect.
A 63 C -> 71 C shift alone can produce the 3% that would be called a regression. So the
tool must **refuse incomparable pairs**: same chip serial, same driver, same
`tlaImplementationSha256`, temperature within a band, power within a band, and
`noiseExceedsEffect` false. Refusal is a first-class outcome, not an error.

This is deliberately **not** a gate. The harness has no baseline concept, and inventing one
inside a deterministic-test gate would mean encoding a threshold the evaluator cannot
check - a fail-open in waiting.

### P5 - Pin the wheel by content, not by name

`TLA_NPU_TRITON_WHEEL` pins a path. A wheel rebuilt at the same version with different
contents passes the pin. Record the wheel file's `sha256` in the receipt and have
`setup-python-env.sh` refuse a wheel whose hash changes between provisioning and a later
run unless the operator re-provisions deliberately.

Lowest priority of the six items, but it closes the last provenance gap: today the receipt
names the *installed tla files*, which is the right thing, but cannot say which wheel file
produced them.

## 6. Order and effort

| Phase | Value for the stated goal | Core changes | Blocker |
|---|---|---|---|
| **P-0a** | test dir in receipt identity - makes every later PASS falsifiable | `engine.py` + adapter | none |
| **P-0b** | Hash the test bytes as an observation | adapter only | none |
| **P-0c** | Measurement methodology (median, ms, per-kernel warm, no exclusivity claim) | adapter only | none |
| P0 | Agent's own kernels, enumerated not discovered | adapter only | after P-0a/P-0b |
| P1 | Regression net | adapter only | `run` stage: upstream numerical failures - gate `ir` only |
| P3 | Observations visible through MCP, projected allowlist | `api_types.py`, `mcp_server.py` | none |
| P4 | Comparability-checked compare tool | adapter only | after P-0c |
| P5 | wheel pinned by content | adapter only | none |

P0-P2 are adapter-local and can land without touching the harness core. P3 is the only
item that needs core review, and it is independent of the rest.

**First slice: revised after review, and the revision is the important part.**

My draft said P0 + P2. The review called that the worst-value pairing, correctly: P0 is the
largest and least fail-closed item, and P2 depends on an isolation the adapter cannot
provide. Together they ship "can run, can measure, neither trustworthy".

The review's ordering, adopted:

**P-0a - Give the test directory its own identity in the engine.** The equivalent of
`sourceAfter != sourceBefore`, for the test dir. One function, closes the hole that makes
everything else unfalsifiable. This is the single highest-value change in the whole plan.

**P-0b - Hash the test bytes today, as an observation.** For the single `test_tla.py` that
already exists. No discovery mechanism needed. Once case bytes are in the receipt, P0
becomes mechanical rather than risky.

**P-0c - Then P2 methodology only.** Explicit `warmup`/`rep` in ms, `median`, per-kernel
warming, published `ratioSpread`, no exclusivity claim, gated on noise rather than on the
ratio.

That closes two real fail-closed holes for roughly a fifth of the cost of P0+P2, and needs
no core review.

## 7. Review findings, and what changed

An adversarial architecture review was run against this plan with the evaluator, manifest,
engine, `api_types`, `mcp_server` and live receipt available for verification. Findings,
each re-verified by me before acceptance:

| Finding | Verified how | Status |
|---|---|---|
| Receipt does not bind the test bytes at all (`agent_tla` 0 matches; test sha256 0 matches) | grepped the live PASS receipt | **Confirmed, severe** - new P-0a/P-0b, P0 blocked |
| Manifest accepts a float `observationChecks` value, so "cannot express a threshold" was wrong wording | loaded a patched manifest: `LOADED OK` | Confirmed - wording corrected to a deliberate refusal |
| `do_bench` warmup/rep are milliseconds and default statistic is `mean` | read `testing.py:127-170` | Confirmed - P2 methodology rewritten |
| Only the first kernel is warmed by `do_bench`'s estimate call | read the estimate path | Confirmed - P2 requires per-kernel warming |
| Exclusive device is not enforceable; lease is cooperative among AgentGates runs only | read `require_device_free` and the `npu:<id>` lease | Confirmed - exclusivity claim dropped |
| Name-pinned `expectedFailures` rots | parametrised pytest ids; also could not reproduce my "21" cleanly | Confirmed - P1 now compares a failure-set hash |
| `strict_json` bounds size but not depth/length/count | read `strict_json` | Confirmed - P3 now a named scalar allowlist |
| A ratio from two receipts is tautological without comparability checks | reasoning | Accepted - P4 must refuse incomparable pairs |
| P0+P2 is the worst first slice | judgement | Accepted - reordered to P-0a/P-0b/P-0c |

The review's bottom line, which this document now adopts: **do not implement P0 before
`test_dir` is in receipt identity.** That omission is what would make the rest of the plan
unfalsifiable.

## 8. What this plan deliberately does not do

- No performance threshold in a gate. The evaluator cannot check one today, and this plan
  refuses the operator that would let it, on purpose.
- No baseline stored as an absolute time to compare against. Cross-process drift is 25.8%.
- No "PASS means faster". A PASS means correct, and well-measured.
- No new `gateType`, no new evaluator operator. Any such change would need its own review
  and would put a comparison rule into the one place the whole harness trusts to be exact.
- No editing of the vendor suite. It is an input, run as it ships.