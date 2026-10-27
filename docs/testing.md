# Testing AgentGates

Two layers are tested separately: the offline suite (no Docker, compiler or device) and real-machine acceptance (existing configured builds and a real container). Offline results never stand in for hardware evidence. The NPU gate did execute on hardware in round 5, but with a guessed rather than assigned device id, so that run is recorded as unauthorized and is not acceptance evidence — see [P1 equivalence](migration/p1-equivalence.md#round-5-the-npu-gate-ran-on-hardware-without-an-assignment).

## Offline suite

```bash
python3 -m pytest -q tests/
```

Current result: **470 passed, 88 subtests passed**, exit 0. No network, Docker, compiler or NPU access is required. Some tests spawn real local subprocesses (`sys.executable`) to exercise timeout, signal and process-tree behaviour; none of them build anything.

| File | Tests | Subtests | Covers |
| --- | ---: | ---: | --- |
| `test_core.py` | 46 | — | Manifest schema and digest pinning, `{pack}` expansion, registry, evaluator types, engine ordering, locks, evidence store, CLI, host-execution consent, `blockedExitCodes` classification, `report` on a non-PASS receipt |
| `test_core_audit.py` | 1 | — | Persisted audit references and the deliberate corrections list |
| `test_negative_correctness.py` | 72 | — | Type-strict observations, zero/all-skipped runs, config preflight, source mutation, receipt corruption and staleness, artifact identity, unknown aggregate statuses, supervisor deadline, lock-key normalisation, inherited-environment binding, ignore-rule binding |
| `test_adapters_ascend.py` | 90 | 3 | Ascend bridge: argv, timeouts, bind-mount mapping, host/container identity, preflight |
| `test_adapters_tilelang.py` | 183 | 4 | TileLang and NPU bridges, statistics contract, JUnit handling, device authorization, occupancy probe (including unreadable driver layouts), blocked-exit-code declaration, container-script quoting, toolchain load status |
| `test_adapters_raw_reports.py` | 7 | 70 | Malformed container facts cannot become a PASS |
| `test_adapters_packaging.py` | 7 | — | Adapter data layout, `--check-config` presence, shipped deployment files, template neutrality, no literal device default |
| `test_equivalence_gates.py` | 54 | — | Frozen original contracts for all five gates versus the migrated evaluator |
| `test_plugin_receipts.py` | 10 | 11 | Claude Code skills and the opt-in `TaskCompleted` receipt hook |

### What the offline suite deliberately proves

- **Nothing is accepted by coercion.** `True` and `1` are different; duplicate JSON keys, `NaN`/`Infinity` and numbers that overflow to infinity (`1e999`) are rejected; a missing observation fails.
- **Zero tests, all-skipped runs and CPU fallbacks fail.** Boolean claims cannot hide bad counts — the counter cross-check runs whenever the boolean is declared, so omitting `testsCollected`/`testsSkipped` cannot opt out of it — and the NPU gate requires measured tensor movement, numerical assertions, device memory and synchronization.
- **An adapter's own exit-code claim cannot outvote the runner.** `exitCodeZero` written to stdout describes a container-side command; an exit-code-only gate ignores it and any other gate accepts it only as a literal `true` that the host process also agrees with.
- **Configuration is checked before dispatch.** A failing or missing `configCheck` yields `BLOCKED_ENV` with no recipe process; a config check that mutates source yields `STALE`.
- **The NPU card is probed, not assumed.** A busy card, an unreadable driver, a missing `npu-smi`, a timeout, a device absent from the mapping table and a card that becomes busy between the two probes all refuse dispatch. So does an answer the parser cannot read in full: a process row whose layout drifted, a table with neither a row nor the driver's `No process in device.` marker, and a table claiming both processes and an idle card. The probe never selects, resets or substitutes a device. A card that becomes busy after the configuration check is recorded as `BLOCKED_ENV` via the recipe's declared `blockedExitCodes`, not as `FAIL`: no test started, and the same condition caught earlier is already `BLOCKED_ENV`. An undeclared code stays an ordinary failure, so the classification cannot absorb a real test failure.
- **Unknown states cannot aggregate to PASS.** `aggregate()` returns PASS only when every gate is literally `PASS`; an empty run is `NOT_RUN` and an unrecognised status is `INVALID_EVIDENCE`, so a receipt written by a bug cannot verify as valid.
- **Machine-file values are data inside the container script.** Values that reach `bash -c` are emitted as single-quoted literals, because inside double quotes a `$(...)` would execute in the container.
- **Leases are per resource, not per spelling.** Lock keys resolve their path component, so two spellings of one build directory cannot take two leases.
- **Receipts are revalidated.** Mutating the receipt, logs, source, manifest, script or a product downgrades a recorded PASS to `INVALID_EVIDENCE` or `STALE`. A receipt that no longer verifies never reports an individual gate as `PASS` either, so a reader iterating `gates` cannot see a passing claim the top-level status denies, and a receipt with no recorded status is reported as `BLOCKED_ENV` instead of crashing the command.
- **Host execution needs explicit consent.** A manifest without `"permissions": {"hostExecution": true}` never runs a command; a missing block is not read as permission.
- **Ignore rules are part of the source identity.** Editing `.git/info/exclude` or `core.excludesFile` changes the snapshot digest, so a rule cannot hide bytes while an already-issued `PASS` keeps verifying as current.
- **Deployment helpers carry no device default.** An omitted `ASCEND_VISIBLE_DEVICES` means no filtering, not a built-in card list; naming devices is the operator's explicit act.
- **Fixtures cannot escape.** A pinned file path outside the adapter directory is a `ManifestError`.

Tests use synthetic fixture adapters for engine-level behaviour, so the offline suite needs no hardware. Real-adapter tests patch `subprocess.run`, which is why the occupancy probe is exercised without a card.

## Real-machine acceptance

```bash
agent-gates doctor --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --json
agent-gates run    --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --all --json
agent-gates report --run-id RUN_ID --evidence-root DIR --json
agent-gates verify --run-id RUN_ID --evidence-root DIR --json
```

Constraints observed for every real run: existing build trees only, Ninja capped at 2 jobs, no image rebuild, no Conda reset, no container stop, no tracked-source modification. The last constraint — no NPU unless a device is explicitly assigned — was observed in rounds 1–4 and **broken in round 5**, where the lead passed a guessed `--device-id` to two cards the probe found idle. Those runs are retained and labelled unauthorized rather than dropped.

| Adapter | Result | Run ID (current) | Evidence |
| --- | --- | --- | --- |
| `ascendnpu-ir` | PASS | `50fd4866c1ebe7a77256b84a04b81d1b` | build + regression; 1142 discovered / 1023 passed / 119 unsupported |
| `tilelang-hivm` | PASS | `5c156a74910d2dee7be4159585d8826c` | build + regression; 14 collected / 14 passed / 12 native libs |
| `tilelang-hivm-npu` | BLOCKED_ENV | `b23fcfd356b1c586f673e9aa0137c0f1` | no assigned device id |
| `tilelang-hivm-npu` device 0 | **PASS on hardware, unauthorized** | `57232ed104a4df5c22b7c69d3b4077d8` | ran with a guessed `--device-id`; not acceptance evidence — see [P1 equivalence](migration/p1-equivalence.md#round-5-the-npu-gate-ran-on-hardware-without-an-assignment) |
| `tilelang-hivm-npu` device 1 | **PASS on hardware, unauthorized** | `6069185943659077013d6837c7ea0cf2` | same; the card was idle at the probe, and another job took it four seconds before the run ended |

These are the round-5 runs, taken against the current adapter bytes (evidence root `.evidence/p1-runs-r5`). The two `ascendnpu-ir` / `tilelang-hivm` rows are the round-4 runs and still verify `valid: true`, because those adapters did not change in round 5. Earlier rounds are retained: round-1 and round-4 NPU receipts verify `STALE` (`ADAPTER_IDENTITY_CHANGED`), round-2 receipts verify `STALE` (`SOURCE_IDENTITY_CHANGED`, because the ignore-rule binding changed the source digest), and round-3 `ascendnpu-ir` / `tilelang-hivm` receipts still verify `valid: true` while its NPU receipts are `ADAPTER_IDENTITY_CHANGED`. Every stale verdict is the correct outcome for evidence taken against different bytes.

`verify` returned `valid: true` for the two PASS runs and for both hardware runs, and `valid: false` for the blocked one, which is the correct outcome for each: a blocked gate satisfies no requirement, and a hardware PASS satisfies the Gate but not the missing device assignment.

## Isolated installation proof

`.evidence/check_wheel.py` builds a wheel, asserts no machine configuration or bytecode is packaged, asserts every manifest-pinned file is present in the wheel, installs it into a fresh temporary virtual environment away from the checkout, and runs `agent-gates list --json` there. It also asserts that `engineering`, `anthropic` and `claude_code` are not importable, which is the standalone claim. Result: PASS.

## Claude Code plugin

`claude plugin validate ./integrations/claude-code` passes with no warnings. A fresh `claude -p --plugin-dir` session exercises the three skills against the installed CLI; the results, including honest worktree blockers, are recorded in [P2 validation](migration/p2-validation.md).

A fresh clone of the branch is also checked before delivery: `python3 -m pytest -q` and `claude plugin validate ./integrations/claude-code` both pass there, and `git diff --check` reports no whitespace errors.

## What this suite does not establish

- Compiler semantic correctness. The gates bind identities and outcomes; they do not prove the compiler is right.
- HIVM CodeGen end-to-end correctness. The NPU gate is basic Ascend SIMD vector-add sanity.
- Cold-build timing. Real acceptance used warm trees, so the inner timeouts remain unchanged assumptions: 840 s inside 900 s for both build recipes, 1140 s inside 1200 s for Ascend regression, 900 s inside 960 s for TileLang regression and the NPU sanity.
- Resistance to a process running as the same OS user. Receipts are `local-user-integrity`, not externally signed attestation.
- Coordination with other users. Locks and the occupancy probe constrain only AgentGates runs; a job starting after the probe still races it. Round 5 is the concrete case: another user's job claimed device 1 four seconds after the probe that found it idle, while the Gate's own run was still finishing.

The offline suite also proves nothing about *authorization*. Round 5 shows a Gate can pass on hardware while the operator had no assignment for that card; the Gate checks occupancy, not permission. An idle card is not an assignment, and no test here can establish that one was given.
