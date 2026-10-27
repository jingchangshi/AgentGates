# AgentGates V2 — Test Report

Every result below is marked PASS, FAIL, BLOCKED or NOT_TESTED, and separated by
kind. A unit test and a real compiler run are not the same evidence, and this
report never lets one stand in for the other.

## Summary

| Area | Result | Kind | Evidence |
|---|---|---|---|
| AgentGates existing suite (regression) | **PASS** 555 | Unit + Integration | `pytest tests -q` |
| AgentGates V1 baseline preserved | **PASS** 470 | Unit | same run, unchanged tests |
| Config + API contract | **PASS** 37 | Unit | `test_config_and_api.py` |
| Requirement verification | **PASS** 15 | Unit + Integration | `test_verify_requirements.py` |
| Execution semantics | **PASS** 25 | Integration | `test_execution.py` |
| MCP protocol | **PASS** 8 | Integration (real stdio client) | `test_mcp_server.py` |
| Real systemd worker lifecycle | **PASS** | Integration (real user unit) | `tests/live_systemd_e2e.py` |
| Real compiler: TileLang HIVM build | **PASS** | **Real Compiler Gate** | run `13afbcbb…` |
| Real compiler: TileLang HIVM regression | **PASS** | **Real Compiler Gate** | run `8866f8ff…` |
| Real compiler: AscendNPU-IR build | **PASS** | **Real Compiler Gate** | run `da89fdd1…` |
| Real compiler: AscendNPU-IR regression | **PASS** | **Real Compiler Gate** | run `098aa676…` |
| Real NPU gate | **PASS** | **Real NPU Gate** | run `d7a13bca…` on a shared device 0 |
| Team completion gate (DSH domain) | **PASS** 136 | Unit + Integration | vitest |
| AgentGates policy vs real verifier | **PASS** 23/23 | Integration (real receipts) | `policy.mjs` |
| MCP bridge preset scoping | **PASS** 30/30 | Integration (real profile boot) | `tool-scope.mjs` |
| Restart / crash recovery | **PASS** | Integration (real worker) | `live_systemd_e2e.py`, recovery proof |
| Full scripted Team E2E (all P7 cases) | **NOT_TESTED** | — | no LLM credential on this machine |

## 1. AgentGates unit and integration tests

```
$ .venv/bin/python -m pytest tests -q
555 passed, 88 subtests passed
```

555 = 470 pre-existing (unchanged, still green) + 85 new (37 config/API + 15 requirement + 25 execution + 8 MCP).

What the new tests pin:

- **Config**: every refusal names its dotted field; symlinked components, `..`
  escapes, relative paths, missing files, boolean concurrency and quoted device
  ids are all refused; the digest is stable across equivalent files and ignores
  the evidence/state location.
- **API contract**: `ExecutionRequest` has exactly four fields, so a machine
  parameter has nowhere to be accepted; `BLOCKED_ENV` is a gate status and never
  an error code; every declared error code maps to a family.
- **Requirement verification**: all eight E3 checks, including no-run-id (refused,
  no `latest` fallback), wrong run id, wrong adapter, wrong workspace, changed
  source (STALE), tampered receipt, gate outside the workspace allowlist, and a
  doctor PASS failing to satisfy a requirement.
- **Execution**: submit returns in under 5 s and persists before starting;
  unusable systemd refuses rather than degrading; the concurrency limit refuses
  rather than queueing; a terminal state cannot be overwritten; cancel is
  idempotent; a vanished worker becomes INTERRUPTED; a stale pid from another
  boot is not treated as alive; wait is bounded and clamps an enormous timeout.
- **MCP**: the real server over a real stdio client — ten tools with input *and*
  output schemas, structured content, and refusals that are errors rather than
  success-shaped objects.

## 2. Real systemd worker

`tests/live_systemd_e2e.py`, against the real user manager on this machine:

| Measurement | Result |
|---|---|
| `systemctl --user is-system-running` | `running` |
| submit → return | 0.08–0.37 s (never waits for the build) |
| gate executed by the worker | yes, real subprocess, exit 0 |
| receipt published and verified | `PASS` |
| idempotent re-submit | same execution id returned |
| different request, same id | `CONFLICT` |
| cancel a RUNNING gate | `CANCELLED`, child pid gone, no completion marker, no PASS |
| repeat cancel | same terminal state |

## 3. Real compiler gates — the acceptance evidence

Submitted through `ExecutionManager` and executed by
`agent-gates-worker@<id>.service`. Each receipt was then re-verified and passed
to `verify_requirements`.

### TileLang HIVM — build **(PASS)**

```
run     13afbcbbc51058b118b951e1a12d8274
adapter tilelang-hivm   manifest 1669e8632062b67b
source  /home/s00653124/workspace/tilelang
gate    tilelang-hivm-build  PASS  exit 0
verify  PASS []
verify_requirements  satisfied=true  matched=[tilelang-hivm-build]
```

### TileLang HIVM — regression **(PASS)**

```
run     8866f8ff996362209cc8574cb33ec1ea
gates   tilelang-hivm-build PASS, tilelang-hivm-regression PASS
observations  testsCollected=14  testsSkipped=0  notAllSkipped=true
              importedModuleSha256MatchesHost=true
              loadedNativeSha256MatchesHost=true
              nativeLibsPresent=true  exitCodeZero=true
verify  PASS []
verify_requirements  satisfied=true  matched=[build, regression]
```

14 real tests collected, none skipped, and the loaded native library was
confirmed byte-identical to the host file. This is a genuine compiler
regression, not a smoke test.

### AscendNPU-IR — build **(PASS)**

```
run     da89fdd123442b5c2f1c7a31627b7435
adapter ascendnpu-ir
gate    ascend-build  PASS  exit 0
verify  PASS []
verify_requirements  satisfied=true
```

### AscendNPU-IR — regression **(PASS)**

```
run     098aa6768c62fd92bd8b53a995e66ef8
gates   ascend-build PASS, ascend-regression PASS
output  Total Discovered Tests: 1142
        Passed: 1009 (88.35%)   Unsupported: 133 (11.65%)   Failed: 0
wall    2.7 s for the regression recipe, 0.8 s for the build (tree already built)
verify  PASS []
verify_requirements  satisfied=true  matched=[ascend-build, ascend-regression]
```

**A correction worth recording.** An earlier reconnaissance estimated this suite
at 20–60+ minutes and on that basis the gate was left unrun. The first real run
disproved it: lit drives 1142 tests across 256 workers and finishes in under two
seconds of testing time. The estimate has been corrected everywhere it appeared,
and the recipe timeout remains raised only as deliberate slack for an unbuilt
tree — the original 1140 s would have sufficed on a warm one.

`Unsupported: 133` is normal for lit: those tests declare requirements this
platform does not meet. They are neither passes nor failures, and `Failed: 0` is
the number that decides the verdict.

### TileLang NPU **(PASS — real device, shared card)**

```
run     d7a13bcaf059d39aeafdff2670c3cd71
adapter tilelang-hivm-npu
gate    tilelang-hivm-npu-sanity  PASS  exit 0  elapsed 9.0s
config  device=0 npuId=0 occupancy=shared(6) exclusive=False
observations
  npuDeviceUsed              = true
  deviceMatchesAuthorized    = true
  tensorMovedToDevice        = true
  synchronizeOk              = true
  numericalAssertionsObserved= true
  testsCollected             = 1     allPassed = true     notAllSkipped = true
verify  PASS []
verify_requirements  satisfied=true  matched=[tilelang-hivm-npu-sanity]
```

This ran on **device 0 while six other processes held the same card**. That is
the intended policy: sharing an NPU is normal, and a gate that refused because
someone else was on the device would be the only workload on this host that
could not run. The receipt still proves what an NPU verdict must prove — the
tensor moved to the device, the device reported by the driver equals the one
requested, the synchronisation succeeded, and the numerical assertions actually
ran (`testsCollected = 1`, so this cannot be a quietly-skipped test).

The configuration check reports `occupancy=shared(6)` so a reader can tell a
contended run from a quiet one; `devices.require_exclusive_device = true` turns
the same situation into `BLOCKED_ENV` for runs where that matters.

## 4. Team completion gate

| Layer | Result |
|---|---|
| DSH domain (136 tests, 100% per-file coverage of `agent-team/src`) | **PASS** |
| Bypass audit — every `status: completed` write | two lines, both fenced |
| Replay of a foreign log with a forged completion | rejected (terminal projection failure) |
| AgentGates policy against real receipts (23 checks) | **PASS** |
| Real Team service + policy + CLI (5 checks) | **PASS (partial coverage)** |
| Full scripted scenario with every P7 negative case | **NOT_TESTED** |

The `NOT_TESTED` item is blocked by a machine constraint, not by a defect: this
host has no LLM credential, and a DSH profile treats a credential-less provider
as a fatal boot condition. `docs/11-team-gate-e2e-scope.md` in the
`dsh-compiler-mode` repository states exactly which claims are proven and how to
finish the run elsewhere.

## 5. MCP bridge scoping — measured

`node tests/mcp/tool-scope.mjs` → 30/30. The numbers matter as much as the pass:

| Preset | AgentGates tools resolved | PTC SDK bindings |
|---|---|---|
| `ai-compiler` | 10/10 | 10/10 |
| `ai-compiler-magpie` | 10/10 | 10/10 |
| `standard` | 0/10 | 0/10 |
| `ptc` | 0/10 | 0/10 |
| `minimal` | 0/10 | 0/10 |
| `cordis` | 0/10 | 0/10 |

| Resource measurement | Value |
|---|---|
| MCP child processes at boot | 2 (one per compiler preset), before any Agent |
| Additional children per Agent | 0 |
| RSS per child | ~67.5 MiB |
| Boot to ready | 1.29–1.73 s |
| Agent creation | 8–17 ms |

**Measured correction to the initial design assumption:** the bridge is per
*preset*, not per *Agent*, because presets mount eagerly at registration. An
8-member Team therefore holds 2 processes, not 8 — which is why a stdio server
per preset is acceptable and a shared service is unnecessary. It also means a
Standard-mode session pays for both compiler bridges while the bundle is
installed.

## 6. Restart and crash recovery

Against the real worker and the real store:

| Scenario | Result |
|---|---|
| Manager process discarded, fresh manager reads state | correct state reported |
| Idempotent re-submit after the restart | same execution id |
| Worker killed with no receipt | `INTERRUPTED`, never `PASS` |
| Receipt written, status file rolled back to `RUNNING` | receipt adopted, `FINISHED`, verifies `PASS` |
| Stale pid from another boot | not treated as alive |
| Terminal state vs late event | terminal state wins |

## 7. What this report does NOT claim

- No NPU result.
- No AscendNPU-IR regression result.
- No full model-driven Team scenario.
- The HMAC is a local user-level integrity check, not a signature. It detects
  accidental modification and evidence that disagrees with the execution record.
  It does **not** defend against malicious code running as the same OS user, and
  it is not remote attestation or hardware proof.
