# P0 extraction audit

Audited baseline: PiEngineeringHarness `9ca04ba6814154a778ecf30168c793d73352e643`. The independent validator read the implementation and original manifests, and ran the original Ascend/TileLang adapter mock suites read-only: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider <Harness>/tests/core/test_ascendnpu_ir_adapter.py <Harness>/tests/core/test_tilelang_hivm_adapter.py`, exit 0, **264 passed plus 3 subtests**, 1.16 seconds. These are offline tests, not device execution.

| Original boundary | Actual behavior and extraction decision |
| --- | --- |
| `engineering/packs.py` | Manifest file pins are checked before use. Preserve content hashes and reject missing/changed files. |
| `engineering/runner.py` | Resolve fixed recipe argv and source/pack CWD; expand `{pack}` against the validated pack. Preserve these semantics without task orchestration. |
| `engineering/preflight.py` | Run declared configCheck with bounded timeout before execution; nonzero blocks work. Bundled migrated recipes all declare a pinned check, including a new read-only Ascend check. |
| `engineering/execution.py` | Derive exitCodeZero from process result. Exact exit-code-only rules accept ordinary build stdout. Structured gates parse the entire bounded stdout as a JSON object. Preserve valid acceptance conditions; reject malformed types more strictly. |
| `engineering/provenance.py` | Bind source, recipe/executable and files/environment to evidence. Retain local integrity checks and document same-user trust limitations. |
| `engineering/workspace.py` | Task/workspace state supplies source identity. Replace with direct Git-root snapshots and explicit CLI source selection. |
| `engineering/supervision.py` | Full task/session supervisor coupling is outside extraction scope. Retain only required execution lifecycle behavior. |
| `engineering/process_supervisor.py` | Timeout/signal cleanup and child supervision inform the standalone process boundary. Docker work still requires its own container timeout. |
| Original receipts and PASS reuse | Original reuse depends on task/writer/dependency/source/config/executable identities. AgentGates never substitutes a historical PASS for a new run. |

The original evaluator used Python equality (`observed.get(key) == expected`) in its rule comparison. This accepts `True == 1`. AgentGates deliberately requires exact JSON value types and rejects duplicate keys/nonfinite numbers. The original NPU bridge also normalized malformed raw facts into apparently valid observations; independent negative tests reproduce that issue. Fixes are acceptance hardening, not a reduction of the original valid-input conditions. See [gate semantics](gate-semantics.md) and [Independent review](progress.md#independent-review) for resolved cases.

The extraction excludes StateStore/TaskSpec, owner approval, agent sessions, model routing and orchestration. The CLI directly selects an adapter, source and gate set. Compiler algorithms, pass semantics and test selection remain owned by the compiler projects.

Original recipe execution is documented separately in [P0 baseline](p0-baseline.md). It passed four CPU/build gates, left NPU blocked for absent assignment, and has an explicit gap in retained per-native-library P0 hash evidence. Offline fixtures are synthetic contract inputs and are never represented as real A5 results.
