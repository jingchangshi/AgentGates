# Gate semantics and extraction boundaries

Audited source: PiEngineeringHarness `9ca04ba6814154a778ecf30168c793d73352e643`. Original manifests and an import-free legacy evaluator fixture are retained in `tests/fixtures/legacy/` and `tests/fixtures/legacy_oracle.py`; those fixtures support offline comparisons without importing the Harness at runtime.

| Gate | Fixed recipe action | Outer/inner timeout (seconds) | Required observations | Dependency |
| --- | --- | --- | --- | --- |
| `ascend-build` | Three Ninja targets: `bishengir-opt`, `bishengir-compile`, `install-bishengir-publish-products` | 900 / 840 | `exitCodeZero` | none |
| `ascend-regression` | Ninja `check-bishengir` | 1200 / 1140 | `exitCodeZero` | `ascend-build` |
| `tilelang-hivm-build` | Ninja `tilelang` | 900 / 840 | `exitCodeZero` | none |
| `tilelang-hivm-regression` | Fixed two-file Ascend host pytest suite | 960 / 900 | imported module/native hashes match host; native libraries present; tests collected; not all skipped | `tilelang-hivm-build` |
| `tilelang-hivm-npu-sanity` | Fixed basic Ascend SIMD sanity suite | 960 / 900 | allPassed; authorized device match; tests collected; not all skipped; device used; numerical assertions; synchronization; tensor transfer | none |

The inner `timeout` command retains a 15-second kill grace. The outer recipe timeout exceeds that envelope. Nonzero exit, timeout and interruption always fail independently of an observation claim. Plain Ninja stdout is valid for an exit-code-only gate; structured-observation gates require a valid JSON object. Required observations must match both type and value; integer 1 cannot replace boolean true.

The old TileLang manifests have `configCheck` commands. The old Ascend manifest has none; the migrated Ascend bridge adds a read-only `--check-config` and declares it for both recipes. This strengthens preflight without changing the Ninja targets. Machine files are parsed as strict allowlisted assignments, not sourced. Existing reviewed CANN/Conda initialization scripts still run inside the selected container.

`{pack}` names the installed adapter directory. AgentGates expands only references declared in the manifest `files` digest map; modifying a script requires updating its manifest digest. Docker bridge source mapping is derived from actual bind mounts and verified using host/container directory device/inode identity. Neither a matching path spelling nor a container name alone proves the source tree.

TileLang regression checks module and native library bytes against host files. Its fixed test selection is `testing/ascend/target/test_tilelang_ascend_target.py` and `testing/ascend/cache/test_tilelang_ascend_binary_cache.py`. The NPU gate is a separate adapter and requires explicit assignment. It is not included by `tilelang-hivm --all`, and it does not prove whole-backend E2E correctness.

The extracted CLI replaces Harness StateStore/TaskSpec-dependent orchestration with a direct gate DAG and per-run receipts. It preserves fixed command execution, process timeouts, deterministic observation checks and artifact provenance. It intentionally omits Harness agent scheduling, owner/task approvals, sessions, model routing and communication bridges. Local receipts bind current identities and do not serve as a historical PASS cache.

Documented hardening must be supported by negative tests. Any confirmed original false PASS should be corrected and recorded in the equivalence report, rather than reproduced merely to match the legacy fixture.
