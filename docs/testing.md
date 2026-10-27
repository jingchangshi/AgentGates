# Testing AgentGates

Two layers are tested separately: the offline suite (no Docker, compiler or device) and real-machine acceptance (existing configured builds and a real container). Offline results never stand in for hardware evidence, and the NPU gate has no real execution evidence on this machine — see [P1 equivalence](migration/p1-equivalence.md).

## Offline suite

```bash
python3 -m pytest -q tests/
```

Current result: **452 passed, 88 subtests passed**, exit 0. No network, Docker, compiler or NPU access is required. Some tests spawn real local subprocesses (`sys.executable`) to exercise timeout, signal and process-tree behaviour; none of them build anything.

| File | Tests | Subtests | Covers |
| --- | ---: | ---: | --- |
| `test_core.py` | 33 | — | Manifest schema and digest pinning, `{pack}` expansion, registry, evaluator types, engine ordering, locks, evidence store, CLI |
| `test_core_audit.py` | 1 | — | Persisted audit references and the deliberate corrections list |
| `test_negative_correctness.py` | 71 | — | Type-strict observations, zero/all-skipped runs, config preflight, source mutation, receipt corruption and staleness, artifact identity, unknown aggregate statuses, supervisor deadline, lock-key normalisation, inherited-environment binding |
| `test_adapters_ascend.py` | 90 | 3 | Ascend bridge: argv, timeouts, bind-mount mapping, host/container identity, preflight |
| `test_adapters_tilelang.py` | 181 | 4 | TileLang and NPU bridges, statistics contract, JUnit handling, device authorization, occupancy probe (including unreadable driver layouts), container-script quoting |
| `test_adapters_raw_reports.py` | 7 | 70 | Malformed container facts cannot become a PASS |
| `test_adapters_packaging.py` | 5 | — | Adapter data layout, `--check-config` presence, shipped deployment files, template neutrality |
| `test_equivalence_gates.py` | 54 | — | Frozen original contracts for all five gates versus the migrated evaluator |
| `test_plugin_receipts.py` | 10 | 11 | Claude Code skills and the opt-in `TaskCompleted` receipt hook |

### What the offline suite deliberately proves

- **Nothing is accepted by coercion.** `True` and `1` are different; duplicate JSON keys and `NaN`/`Infinity` are rejected; a missing observation fails.
- **Zero tests, all-skipped runs and CPU fallbacks fail.** Boolean claims cannot hide bad counts, and the NPU gate requires measured tensor movement, numerical assertions, device memory and synchronization.
- **Configuration is checked before dispatch.** A failing or missing `configCheck` yields `BLOCKED_ENV` with no recipe process; a config check that mutates source yields `STALE`.
- **The NPU card is probed, not assumed.** A busy card, an unreadable driver, a missing `npu-smi`, a timeout, a device absent from the mapping table and a card that becomes busy between the two probes all refuse dispatch. So does an answer the parser cannot read in full: a process row whose layout drifted, a table with neither a row nor the driver's `No process in device.` marker, and a table claiming both processes and an idle card. The probe never selects, resets or substitutes a device.
- **Unknown states cannot aggregate to PASS.** `aggregate()` returns PASS only when every gate is literally `PASS`; an empty run is `NOT_RUN` and an unrecognised status is `INVALID_EVIDENCE`, so a receipt written by a bug cannot verify as valid.
- **Machine-file values are data inside the container script.** Values that reach `bash -c` are emitted as single-quoted literals, because inside double quotes a `$(...)` would execute in the container.
- **Leases are per resource, not per spelling.** Lock keys resolve their path component, so two spellings of one build directory cannot take two leases.
- **Receipts are revalidated.** Mutating the receipt, logs, source, manifest, script or a product downgrades a recorded PASS to `INVALID_EVIDENCE` or `STALE`.
- **Fixtures cannot escape.** A pinned file path outside the adapter directory is a `ManifestError`.

Tests use synthetic fixture adapters for engine-level behaviour, so the offline suite needs no hardware. Real-adapter tests patch `subprocess.run`, which is why the occupancy probe is exercised without a card.

## Real-machine acceptance

```bash
agent-gates doctor --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --json
agent-gates run    --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --all --json
agent-gates report --run-id RUN_ID --evidence-root DIR --json
agent-gates verify --run-id RUN_ID --evidence-root DIR --json
```

Constraints observed for every real run: existing build trees only, Ninja capped at 2 jobs, no image rebuild, no Conda reset, no container stop, no tracked-source modification, no NPU unless a device is explicitly assigned.

| Adapter | Result | Run ID (current) | Evidence |
| --- | --- | --- | --- |
| `ascendnpu-ir` | PASS | `52ea3b30495dba9d086cabe0dc9c38bb` | build + regression; 1142 discovered / 1023 passed / 119 unsupported |
| `tilelang-hivm` | PASS | `61ed95d533a314da934e4e8e6cd6eca7` | build + regression; 14 collected / 14 passed / 12 native libs |
| `tilelang-hivm-npu` | BLOCKED_ENV | `e6d1680cd9b2972d645729089cc5a1ae` | no assigned device id |
| `tilelang-hivm-npu` device 0 | BLOCKED_ENV | `57063f7d46de74f1cee9b8ac87c761ff` | card held by another job (7 holders) |
| `tilelang-hivm-npu` device 1 | BLOCKED_ENV | `b767bf002a52e80b721ab1180f927e72` | card held by another job (4 holders) |

These are the round-2 runs, taken after the review fixes changed the adapter scripts; the round-1 receipts are retained under `.evidence/p1-runs` and now verify as `STALE` (`ADAPTER_IDENTITY_CHANGED`), which is the correct outcome for evidence taken against different adapter bytes.

`verify` returned `valid: true` for the two PASS runs and `valid: false` for the blocked ones, which is the correct outcome: a blocked gate satisfies no requirement.

## Isolated installation proof

`.evidence/check_wheel.py` builds a wheel, asserts no machine configuration or bytecode is packaged, asserts every manifest-pinned file is present in the wheel, installs it into a fresh temporary virtual environment away from the checkout, and runs `agent-gates list --json` there. It also asserts that `engineering`, `anthropic` and `claude_code` are not importable, which is the standalone claim. Result: PASS.

## Claude Code plugin

`claude plugin validate ./integrations/claude-code` passes with no warnings. A fresh `claude -p --plugin-dir` session exercises the three skills against the installed CLI; the results, including honest worktree blockers, are recorded in [P2 validation](migration/p2-validation.md).

A fresh clone of the branch is also checked before delivery: `python3 -m pytest -q` and `claude plugin validate ./integrations/claude-code` both pass there, and `git diff --check` reports no whitespace errors.

## What this suite does not establish

- Compiler semantic correctness. The gates bind identities and outcomes; they do not prove the compiler is right.
- HIVM CodeGen end-to-end correctness. The NPU gate is basic Ascend SIMD vector-add sanity.
- Cold-build timing. Real acceptance used warm trees, so the 840 s / 1140 s inner timeouts are unchanged assumptions.
- Resistance to a process running as the same OS user. Receipts are `local-user-integrity`, not externally signed attestation.
- Coordination with other users. Locks and the occupancy probe constrain only AgentGates runs; a job starting after the probe still races it.
