# tilefoundry-npu

The host-local TileFoundry-on-Ascend Adapter (`scripts/tilefoundry_host_gate.py`).

> [!WARNING]
> Legacy container helpers (`scripts/tilefoundry_gate.py`, `scripts/provision.py`)
> are **quarantined** with `DOCKER_MODE_DISABLED` (see
> [`docs/docker-mode-quarantine.md`](../../../docs/docker-mode-quarantine.md)).
> All active TileFoundry validation runs directly on the host through
> `scripts/tilefoundry_host_gate.py`, registered in AgentGates CLI, MCP server,
> and the `systemd` user worker (`tilefoundry-npu`).

## Gates

| Gate ID | Operation | CANN | NPU Lease |
| --- | --- | --- | --- |
| `tilefoundry-npu-cpu` | Conda CPU `pytest` suite (`TORCH_DEVICE_BACKEND_AUTOLOAD=0`, `TILELANG_DISABLE_CACHE=1`, `PYTHONPATH=src:targets/ascend:targets/apple:targets/nvidia`) with JUnit XML verification | **Not sourced** (avoids `ortools`/CANN native library collision) | None |
| `tilefoundry-npu-device` | Depends on `tilefoundry-npu-cpu`; sources CANN `set_env.sh` + Conda `tilefoundry-npu`, runs in-process NPU probe (`import tilefoundry`, FP32 vector add, BF16 Cube `matmul`, memory > 0, `torch.npu.synchronize()`), and optionally runs `TILEFOUNDRY_NPU_DEVICE_TESTS` via `pytest` | Sourced (`TILEFOUNDRY_NPU_CANN_SETUP`) | Windowed (`deviceWindow`: 60 s) |
| `tilefoundry-npu-device-fast` | Same NPU activation and window as `tilefoundry-npu-device`, but runs only the configured `TILEFOUNDRY_NPU_DEVICE_TESTS` (no built-in probe) in legacy fixed mode; when both are empty it exits `BLOCKED_ENV` instead of publishing a zero-test pass. With the typed [`tilefoundry-fast-request/v1`](#device-fast-typed-request) pair it always runs the adapter-owned harness node `tests/ascend/agent/test_surrogate_batch_npu.py::test_surrogate_anchor_batch_npu` for both `batch` and `profile`, regardless of `TILEFOUNDRY_NPU_DEVICE_TESTS`; `profile` requests take the host msprof lock automatically | Sourced | Windowed (`deviceWindow`: 60 s) |
| `tilefoundry-npu-l3-promotion` | L3 promotion profile: one `dynamicTarget` recipe takes a canonical CANNBench task and a source-relative candidate, activates CANN/Conda/TileLang itself, runs a cache/import/device preflight, then executes the whole `tests/ascend/benchmarks/test_npu_acceptance.py` suite under the target interpreter on an authorized (possibly shared) card; records interpreter, module sources, candidate digest, per-phase timings, JUnit rows, reference-cache counters, and coverage | Sourced | Windowed (`deviceWindow`: 60 s) |
| `tilefoundry-npu-cannbench-prep` | Verifies pure-Python CANNBench dependencies (`PyYAML`, `expecttest`, `ruamel.yaml`, `build`, `safetensors`), builds ABI-isolated `cann_bench_utils/_C.so` under `TILEFOUNDRY_NPU_WORK_ROOT`, and verifies `cann_bench_clone` on NPU; takes the device lease and requires an idle card before inspection/build | Sourced | Windowed (`deviceWindow`: 60 s) |
| `tilefoundry-npu-cannbench` | **Local testing mode**: depends on `tilefoundry-npu-cannbench-prep`; takes the host msprof lock, then an idle device lease, injects the `--source` TileFoundry `PYTHONPATH`, runs official `cann-bench` `kernel_eval.staged_eval` against `tasks/<levelN>/<op>`, persists reports under `TILEFOUNDRY_NPU_WORK_ROOT/reports`, and records `score` / `avgSpeedup`; preserves report evidence and phases on failure/timeout. Declares an optional `dynamicTarget`: a managed V2 request narrows the run to its canonical task and workspace-relative candidate, while an unmanaged invocation evaluates the configured targets file | Sourced | Windowed (`deviceWindow`: 60 s) |
| `tilefoundry-npu-cannbench-remote` | **Remote submission mode**: validates `SUB-BEH` / `EVAL-CANN` compliance, packages a compliant `cann_bench` ZIP under `TILEFOUNDRY_NPU_WORK_ROOT/submissions`, and submits via BenchSite MCP (`submit_kernel` per `https://cannbench.com/docs/mcp`) | Not sourced | None (`device`: `none`) |

One-time non-Torch dependency helper (run before first CANNBench gate if `expecttest`/`ruamel.yaml`/`build` are missing):

```bash
bash agent_gates/adapters/tilefoundry-npu/scripts/setup_cannbench_deps.sh
```

## Machine configuration (`env.sh`)

Copy `env.example.sh` to git-ignored `env.sh` beside it (or point
`TILEFOUNDRY_NPU_ENV_FILE` / `--env-file` at an absolute path) and fill in host
paths:

```bash
cd agent_gates/adapters/tilefoundry-npu
cp env.example.sh env.sh
chmod 600 env.sh
```

Only strict `NAME=value` data lines are accepted (no shell expansion, no
conflicting exports in the launching environment):

- `TILEFOUNDRY_NPU_CANN_SETUP`: required absolute host path to CANN `set_env.sh`
  (sourced only in `tilefoundry-npu-device`).
- `TILEFOUNDRY_NPU_CONDA_SH` / `TILEFOUNDRY_NPU_CONDA_ENV` /
  `TILEFOUNDRY_NPU_CONDA_ENVS_PATH` / `TILEFOUNDRY_NPU_PYTHON`: Conda activation
  and Python 3 interpreter containing matched `torch`, `torch_npu`, `pytest`,
  and `z3`.
- `TILEFOUNDRY_NPU_CPU_TESTS`: optional comma- or whitespace-separated **relative**
  pytest paths under `--source` (no leading `/`, no `..`). Leave empty for the
  5 default CPU directories:
  `tests/tilefoundry,tests/ascend/target,tests/ascend/ir,tests/ascend/passes,tests/ascend/operators`.
  The default also appends `tests/ascend/agent`, `tests/ascend/sil`, and
  `tests/ascend/benchmarks/test_l3_contract_host.py` when those paths exist, so
  checkouts without the newer suites keep collecting the original five. The
  five device-only node ids under `tests/ascend/agent` and `tests/ascend/target`
  are passed as exact `--deselect=` arguments when their files exist, so the
  CPU gate keeps the host assertions in those files without opening a device.
- `TILEFOUNDRY_NPU_CPU_WORKERS`: optional integer in `1..16` controlling how
  many parallel file-sharded `pytest` worker subprocesses `tilefoundry-npu-cpu`
  spawns (with single-threaded BLAS/OpenMP/PyTorch env per worker). Leave empty
  for automatic `min(12, cpu_count)`.
- `TILEFOUNDRY_NPU_DEVICE_TESTS`: optional comma- or whitespace-separated
  **relative** pytest paths under `--source` executed inside the NPU lease
  window after the built-in FP32/BF16 probe. Leave empty to run only the
  built-in runtime & compute probe. A typed fast request ignores this key and
  runs only the fixed harness node described below.
- `TILEFOUNDRY_NPU_BUILD_SUBDIR`: single relative directory name under `--source`
  used for AgentGates `build:<dir>` lease (default `build`; never set empty).
- `TILEFOUNDRY_NPU_CANNBENCH_REPO` / `TILEFOUNDRY_NPU_WORK_ROOT` / `TILEFOUNDRY_NPU_CANNBENCH_ARCH`:
  required by CANNBench gates (official `cann-bench` checkout path holding `scripts/run_evaluation.sh`,
  `src/kernel_eval`, `src/cann_bench_utils`, and `tasks/metadata`; isolated host work root outside
  both checkouts for ABI-isolated `cann_bench_utils`, `reports/`, and `submissions/`; and target arch
  defaulting to `ascend950`).
- `TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE` / `TILEFOUNDRY_NPU_CANNBENCH_TARGETS`:
  in-repo JSON target manifest (default `examples/ascend950pr_ops/cannbench_targets.json`) or
  `env.sh` fallback list (`levelN/op:relative/candidate.py`).
- `TILEFOUNDRY_NPU_L3_TIMEOUT_SEC`: optional L3 workload budget in seconds
  (positive, at most 1800; empty means 1200). The recipe's outer bound is 1920
  seconds and its configCheck has an independent bound. The L3 preflight, lock
  wait, compile, and acceptance run all spend from this one monotonic deadline.
  L4 keeps its existing 840/900 prep and 3300/3600 official bounds.
- L3 is a `dynamicTarget` gate: the engine publishes only
  `AGENT_GATES_TARGET` (canonical task such as `level4/attention_residual`) and
  `AGENT_GATES_CANDIDATE` (canonical source-relative `.py` path) to that recipe
  and its configCheck. Never export the pair from a parent shell. The adapter
  independently validates both, binds the candidate bytes by SHA-256, and
  records that digest in the receipt.
- Both L4 recipes (`tilefoundry-npu.cannbench-prep` and
  `tilefoundry-npu.cannbench`) accept the same optional managed pair so prep and
  official measurement stay bound to one target; without a pair they keep the
  managed fixed-target behavior from
  `TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE` / `TILEFOUNDRY_NPU_CANNBENCH_TARGETS`.
- `TILEFOUNDRY_NPU_WORK_ROOT` also holds `<work-root>/l3/cache/<operator>/`
  (`tilelang/`, `reference/`, and a legacy `reference-warm.json`) and
  `<work-root>/l3/runs/<operator>/...`. L3 never writes the shared TileLang or
  CANNBench caches. The ordinary gate always runs with
  `TF_L3_CACHE_EXPECTATION=any`, never reads the marker, and never writes it;
  only an explicit strict cold/warm acceptance may use the marker.
- L4 prep and official measurement take the device lease and require an idle
  card before inspection/build/workload; shared or unknown occupancy exits
  `BLOCKED_ENV` (75) before device work. The msprof host lock is still acquired
  separately (official run) and both waits come from the recipe deadline.
- `TILEFOUNDRY_NPU_CANNBENCH_MCP_URL` / `TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN` /
  `TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE` / `TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION` /
  `TILEFOUNDRY_NPU_CANNBENCH_POOL_ID` / `TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN` /
  `TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT`:
  remote BenchSite MCP submission settings (`https://cannbench.com/docs/mcp`) used by
  `tilefoundry-npu-cannbench-remote`. `INDEPENDENT_ENV_CONSENT` (e.g. `v1`) is never
  set automatically and must only be supplied after explicit user consent.
- `TILEFOUNDRY_NPU_LLM_BACKEND` selects `agy_cli`, `openai_compat`, `claude_cli`,
  `codex_cli`, `custom_cli`, or `mock`.
- `TILEFOUNDRY_NPU_LLM_API_BASE` is the HTTP endpoint base URL.
  `TILEFOUNDRY_NPU_LLM_API_KEY_ENV` names the process environment variable that
  holds the API credential. Store only the variable name in `env.sh`.
- `TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL`, `TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL`,
  `TILEFOUNDRY_NPU_LLM_ESCALATE_MODEL`, and `TILEFOUNDRY_NPU_LLM_REPAIR_MODEL`
  choose model IDs for the four generation roles.
- `TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL` (single-token model ID),
  `TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES` (non-negative integer), and
  `TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC` (comma-separated non-negative finite
  floats) configure 503 capacity retry and fallback model resilience for agent
  LLM calls.
- `TILEFOUNDRY_NPU_LLM_MAX_EFFORT` sets the effort cap.
  `TILEFOUNDRY_NPU_LLM_EFFORT_CAP` is a fallback when `MAX_EFFORT` is unset.
  Set `TILEFOUNDRY_NPU_LLM_PASS_REASONING_EFFORT=1` for an HTTP endpoint that
  accepts `reasoning_effort`.
- Set `TILEFOUNDRY_NPU_LLM_STREAM=1` to stream `openai_compat` chat completions
  using server-sent events. Leave it empty for the default JSON response.
- `TILEFOUNDRY_NPU_LLM_MAX_CONCURRENCY` caps concurrent provider calls.
  `TILEFOUNDRY_NPU_LLM_EXPLOIT_TIMEOUT_SEC` and
  `TILEFOUNDRY_NPU_LLM_REASONING_TIMEOUT_SEC` set time budgets in seconds for
  the exploit and repair roles and for the explore and escalate roles, respectively.
- `TILEFOUNDRY_NPU_LLM_CLI_BIN` selects the CLI executable.
  `TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE` supplies a quoted command template for
  `custom_cli`, such as `"{bin} exec -m {model} {prompt}"`.
- `TILEFOUNDRY_NPU_AGY_BIN`, `TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL`,
  `TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL`, `TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL`,
  `TILEFOUNDRY_NPU_AGY_COMMON_ARGS`, and `TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY`
  are supported by existing `agy_cli` deployments.
- `TILEFOUNDRY_NPU_CANN_SERIES`: optional CANN surface series override
  (empty auto-detects to `"cann_9_3"`, the sole active mainline; `"cann_9_2"` is
  accepted ONLY as an explicit inert placeholder override). Forwarded to
  TileFoundry subprocesses as `TILEFOUNDRY_CANN_SERIES` and recorded with
  `bishengArch: "dav-3510"` in V2 receipt `observations.environment`.
- `TILEFOUNDRY_NPU_TILELANG_REPO` / `TILEFOUNDRY_NPU_CANNBOT_DSL_REPO` /
  `TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO`:
  optional absolute host paths to read-only Ascend 950 reference checkouts forwarded when non-empty.

### Multi-Worktree Zero-Lock Concurrency

AgentGates acquires `source:<dir>` and `build:<dir>/build` locks scoped to
the `--source` directory. When running concurrent multi-NPU optimization
campaigns across cards, create an independent `git worktree` per NPU card and
pass each worktree as `--source`, while sharing one external
`--experience-lake-dir`. That keeps single-tree build locks strict while
reducing cross-card lock contention to `0s`.

The host adapter reads LLM settings directly from `env.sh`. When
`TILEFOUNDRY_NPU_LLM_API_KEY_ENV` is set, it supplies the named credential
from the worker environment to the local device run with extra pytest paths.
The adapter drops other `API_KEY` and `*_API_KEY` variables from child
processes in this configuration. An asynchronous systemd worker starts
independently of the shell that submits the run.

`TILEFOUNDRY_NPU_DEVICE_ID` and `TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE` are
per-run launch/config parameters supplied via `--device-id` or
`~/.config/agent-gates/config.toml`, never stored in `env.sh`.

## Device-fast typed request

`tilefoundry-npu-device-fast` accepts one optional typed request through the
engine-published `dynamicTarget` pair. Submit `--target fast/<request_id>` with
the request JSON as `--candidate`; the engine exports
`AGENT_GATES_TARGET=fast/<request_id>` and
`AGENT_GATES_CANDIDATE=<build>/agent_gates/fast/<request_id>.request.json`,
and the candidate is the dynamicTarget candidate, so the usual receipt
byte/SHA binding and `verify-requirements` re-hash apply. `TF_FAST_REQUEST` is
never read from the parent environment; setting it there is a `BLOCKED_ENV`
error. Without the pair the gate keeps the legacy fixed mode:
`TF_SURROGATE_BATCH_REQUEST` and `TF_MICRO_PROBE_REQUEST` pass through
unchanged, with no typed validation, binding, or additional msprof lock.

`request_id` is 32 lowercase hexadecimal characters (a `uuid4().hex`), must
equal the target suffix and the request filename, and selects every derived
path. The request is a regular file of at most 4 MiB under the configured
build subdirectory, containing one JSON object with no duplicate or unknown
keys. Machine configuration is never read from the request: the working
directory, CANNBench checkout, and caches come from the adapter's own machine
values only.

```json
{
  "schema": "tilefoundry-fast-request/v1",
  "mode": "batch",
  "request_id": "0123456789abcdef0123456789abcdef",
  "response_path": "build/agent_gates/fast/0123456789abcdef0123456789abcdef.response.json",
  "warmup": 3,
  "repeat": 5,
  "require_all_passed": false
}
```

`warmup` and `repeat` are positive integers, `require_all_passed` is a boolean,
and `response_path` must equal the derived response path exactly. The response
file must not exist when the request is validated; its parent directory is
created if needed. `batches` and `msprof_targets` are mutually exclusive: the
selected mode's list must be non-empty and the other list absent or empty.

Every candidate must live under
`<build>/agent_gates/fast/<request_id>/candidates/`; absolute paths, empty or
`.`/`..` segments, backslashes, NUL bytes, symlink components, and paths that
escape `--source` are rejected. Only `tilefoundry-npu-device-fast` reads this
descriptor: the canonical L3 and CANNBench config checks validate their own
canonical target/candidate pair and reject a `fast/<request_id>` target. The
adapter validates the whole request in `--check-config-device` and again before
opening the device window, then snapshots the exact request bytes into the
run's temporary directory and gives the workload:

- `TF_FAST_REQUEST`: the 0400 temporary snapshot (not the workspace file)
- `TF_FAST_SOURCE_ROOT`: the resolved `--source` checkout
- `TILEFOUNDRY_NPU_CANNBENCH_REPO`: the adapter-resolved CANNBench checkout
- `TILELANG_CACHE_DIR` and `CANNBENCH_CACHE_DIR`: the adapter-owned
  `<work_root>/surrogate/{tilelang,reference}` cache roots

`TF_SURROGATE_BATCH_REQUEST` and `TF_MICRO_PROBE_REQUEST` are removed from this
child environment so a typed run cannot silently fall back to another request
channel. The typed run always executes
`tests/ascend/agent/test_surrogate_batch_npu.py::test_surrogate_anchor_batch_npu`
for both modes; the request carries data only and cannot select a pytest node
or command. `--check-config-device` applies the same fixed selection and
requires that harness file to exist.

### `batch` mode

Adds a non-empty `batches` list. Each batch has `operator_name`, `entrypoint`,
a non-empty `anchors` list, and a non-empty `candidates` list. Candidate ids are
unique per operator, not globally; duplicate `operator_name` entries and
duplicate `(operator_name, candidate_id)` pairs are request errors. Every
candidate has a `candidate_id`, a staged `module_path`, and the lowercase
SHA-256 of that module; the declared digest must match the bytes at validation
time. Multiple operators may share one request and one device lease:

```json
{
  "operator_name": "attention_residual",
  "entrypoint": "main",
  "candidates": [
    {
      "candidate_id": "cand_0",
      "module_path": "build/agent_gates/fast/0123456789abcdef0123456789abcdef/candidates/cand_0.py",
      "sha256": "<64 lowercase hex chars>"
    }
  ],
  "anchors": [
    {
      "task": "level4/attention_residual",
      "case_num": 1,
      "regime_tag": "compute",
      "ref_perf_us": 12.5,
      "t_hw_us": 10.0
    }
  ]
}
```

### `profile` mode

Uses `"mode": "profile"` and a non-empty `msprof_targets` list instead of
batches. Each target has `operator_name`, `candidate_id`, a staged
`module_path`, its `sha256`, `case_num`, `warmup`, and `repeat`. One
`(operator, candidate_id)` always names the same module and digest; duplicate
`(operator, candidate_id, case_num)` triples are rejected. Profile mode always
takes the host-wide msprof lock before the device window, without requiring
`TILEFOUNDRY_NPU_MSPROF_LOCK`.

### Response contract

The workload writes one `tilefoundry-fast-response/v1` object to the derived
response path. The envelope echoes `request_id`, `request_sha256`, `mode`, and
`status` (`ok`, `candidate_failure`, `request_error`, or
`infrastructure_error`). `candidates[operator][candidate_id]` must contain
exactly `module_path`, `sha256`, `post_sha256` and match the declared identity;
`coverage` must report nonnegative
`expected_candidates`/`evaluated_candidates` (the declared candidate count) and
`expected_msprof_targets`/`observed_msprof_targets` (the declared profile case
count). Batch responses carry exactly the declared operators and candidates in
`operators`; profile responses carry exactly the declared operators,
candidates, and case numbers in `msprof_telemetry`.

After the workload finishes, the adapter re-hashes the workspace request and
every candidate module and compares them with the pre-run values. A missing,
changed, or replaced request yields `REQUEST_IDENTITY_CHANGED`; a changed
candidate yields `FAST_CANDIDATE_CHANGED:<operator>:<candidate>`. Both, a
missing or mismatched response (`FAST_RESPONSE_UNREADABLE:<reason>`,
`FAST_RESPONSE_REQUEST_ID`, `FAST_RESPONSE_REQUEST_SHA`,
`FAST_RESPONSE_COVERAGE_*`), and a `request_error` or `infrastructure_error`
status are infrastructure blocks: they fail the run and never count as a
candidate regression. Only `status=candidate_failure` from a failing run sets
`failureCategory=candidate`; `ok` runs pass when the configured pytest suite
passes. The request, stable candidates, and response are recorded as gate
artifacts with their SHA-256 digests, and the receipt observations are
`fastRequest`, `requestStable`, `fastCandidates`, `fastResponse`,
`fastCoverage`, `fastBindingFailures`, plus `errorCategory`/`failureCategory`.

Binding diagnostics also include `FAST_RESPONSE_OPERATORS_MISSING`,
`FAST_RESPONSE_TELEMETRY_MISSING`, `FAST_RESPONSE_OPERATOR_MISSING/EXTRA:<operator>`,
`FAST_RESPONSE_CANDIDATE_MISSING/EXTRA:<operator>:<candidate>`, and
`FAST_RESPONSE_CASE_MISSING/EXTRA/INVALID:<operator>:<candidate>[:<case>]`.
Coverage is structural only; the configured pytest suite and the gate's normal
observation checks still decide whether the measured result passes.


## L3 promotion and L4 evidence

The L3 recipe activates once and reuses that activation for its preflight and
the acceptance run. The pinned `scripts/host_control.sh` wrapper starts the
bridge with `-S -E -B` so no project `sitecustomize`, inherited `PYTHON*`
variable, user-site hook, or bytecode cache loads into the control process;
project and benchmark paths are handed only to the target interpreter, whose
`PYTHONPATH` is rebuilt from the TileFoundry source root and the CANNBench
checkout's `src` directory (which provides the official
`kernel_eval.utils.compare` fallback comparator). No `npu_compare` shim or
`sitecustomize` is generated: the suite imports an installed `npu_compare` when
available and otherwise uses the checkout's comparator with unchanged thresholds
and native-reference semantics.
The preflight runs before the device lease and checks, in
the target interpreter, that CANN/NumPy/torch/torch_npu/TileLang/TileFoundry and
the CANNBench comparator import, that torch↔NumPy interop and comparator
semantics work, that both dedicated cache directories accept a write/read/delete
probe, and that the task's `cases.yaml` declares unique case IDs. It records the
interpreter, TileLang effective cache directory, module sources, and build
identity. The workload then runs the whole
`tests/ascend/benchmarks/test_npu_acceptance.py` file under the adapter-published
card; L3 may share an authorized card unless `REQUIRE_EXCLUSIVE_DEVICE` is set.

The acceptance suite writes `tilefoundry-l3-evidence/v1` to
`TF_L3_EVIDENCE_PATH`. The adapter consumes its target/case identity, candidate
digest and module path, interpreter match, reference-cache counters, and
coverage. Promotion requires every expected JUnit test to pass and the evidence
`coverage.qualified` to be true; missing, duplicated, failed, or skipped expected
tests set `coverageComplete=false` and never promote. Unknown test counts stay
`null`; they are never coerced to zero. Phase reports and JUnit rows are kept in
the observation for diagnosis. Only a producer-explicit candidate numeric
comparison failure is a candidate failure: the selected failure must have
`kind="candidate"`, `category="numeric"`, `role="candidate"`,
`stage="comparison"`, a non-empty comparator/comparison result, a node and case
whose evidence status is failed, and a failed JUnit row for that same node. A
bare JUnit `failed` row, a missing producer marker, an unclassified kernel
exception, or a marker whose role/node/case does not line up is never numeric;
those become `coverage`, `environment`, or `infra` blocks instead. Coverage gaps
(missing, duplicated, or skipped expected tests), write errors, environment,
occupancy, deadline, signal, and infrastructure failures restore the original
candidate upstream. The observation keeps the producer `failure`/`failures`
rows and each node's `cases`/`failure` so the classification can be audited.

The ordinary promotion gate accepts `any` cache state, so legal mixed hits,
partial entries, and missing entries that are rebuilt during the run still
promote. The observation still records the signed pre-run cache directories and
tree sizes (`cache.before`), and the producer's per-kind
hits/misses/writes/errors (`referenceCache.byKind`) are preserved verbatim. A
reference-cache write failure remains an environment block.

The receipt observation keeps the evidence summary under `evidence`, including
the producer's nested `target` (`operator`, `task`, `taskDir`, producer-observed
`modulePath`, and case IDs), the producer `candidate` block (`modulePath`,
`sha256`, and `declaredPath` when present), `writeErrors`, interpreter identity,
coverage, and reference-cache hit/miss/write counters. The adapter adds an
engine-signed `candidate.relativePath` from the validated request and a separate
`expectedIdentity` block holding the request-derived target/candidate module
path, relative path, and digest. Producer fields stay unmodified: a missing or
mismatched producer `target.modulePath`, `candidate.modulePath`, or
`candidate.sha256` makes coverage incomplete and blocks promotion rather than
being backfilled from the request.
A top-level `failureCategory` mirrors `errorCategory`; only `numeric` is a
candidate failure. `coverage`, `environment`, `occupancy`, `deadline`, `signal`,
and `infra` are infrastructure blocks that preserve the candidate, so a
promotion client can distinguish them without parsing the adapter's diagnostics.

L4 prep and official runs first take the lease and confirm `free` occupancy. A
shared or unknown card reports `BLOCKED_ENV` before inspection, build, or
workload. The prep and official recipes are optional `dynamicTarget` routes: a
managed V2 request supplies the canonical task and workspace-relative candidate
to both, and the adapter validates that pair itself. The official run then
evaluates exactly that pair (`targetsSource: "dynamic-request"`); without a
request the configured targets file is evaluated unchanged. The official run then takes the host msprof lock,
inspects the isolated `cann_bench_utils` slot, and runs
`kernel_eval.staged_eval`. Its observation records phase start/end, the
monotonic phase durations, the device window, and a snapshot of report files
(including `official_final.json`) even when the runner times out or fails. Prep
and official recipes start their bridge through the same `host_control.sh`
wrapper, so the control process loads no project Python startup hook. The
official workload runs with `-E -s` and rebuilds the target `PYTHONPATH` from
activation-derived CANN paths plus the adapter-known staging, CANNBench,
TileLang, and TileFoundry source paths; inherited project paths and `PYTHON*`
startup hooks never reach the benchmark child. Each L4 target run also exports a
brand-new per-run `TILELANG_CACHE_DIR` under its temporary stage directory with
`TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1` and `TILELANG_DISABLE_CACHE=0`,
preventing cross-run cache contamination while enabling intra-run Stage-2 to
Stage-3 kernel reuse. The official gate still requires
`officialReportValid`, `allCasesPassed`, and `allTimingsValid`; upstream
`verify-requirements` remains responsible for the `satisfied: true` check.

## Runtime health and device-residue supervision

Every L3/L4 host workload runs through `_run_host_bash_monitored`, which keeps
the existing `adapter_runtime.run_host_bash` process-group supervision as the
outer backstop and adds bounded health/occupancy sampling in the adapter. The
workload environment receives an opaque `AGENT_GATES_RUN_HEALTH_TOKEN`; only
processes carrying that token belong to this run and are ever signalled. Other
tenants, including processes sharing an L3 card, are never killed, and the
adapter never resets a card.

| Signal | Budget and action |
| --- | --- |
| `npu-smi info -m`, `npu-smi info -t health -i <npu>`, and `npu-smi info -t proc-mem -i <npu>` probes | Each probe has its own 5 s wall clock, runs in its own session, and is SIGKILLed on timeout. Probes run every 2 s while this run has live processes; one probe never extends the recipe deadline. |
| Probe self-hang or unreapable probe | A probe that survives SIGKILL is tracked by PID and start time. While it exists, no new probe is launched, so a wedged driver cannot accumulate probes. Each suppressed interval counts toward the failure limit; 3 intervals abort with `probe_unavailable` and report `stuckProbePids`. Probe cleanup only signals these adapter-owned probe children, never token/workload descendants. |
| Fatal health value (`ERROR`, `FAULT`, `OFFLINE`, `CRITICAL`, ...) | 2 consecutive probes abort with `device_error`. A single unknown/garbled health row is not a fault. |
| Parsed non-OK health (for example `Warning`) | Recorded as `health="non_ok"`, `nonOkHealthStreak`, and `unknownSignals=["device_health_non_ok"]`; it is not a confirmed fatal fault and does not kill a possibly long kernel. |
| Target card disappears | 2 consecutive `npu-smi info -m` results with no mapping for the requested device abort with `device_missing` (category `environment`, `deviceOccupancy="missing"`). A failed/unreadable mapping command stays `probe_unavailable` instead. Mid-run health failures trigger one bounded mapping recheck so a disappeared card is distinguished from a probe error. |
| Health/mapping probe unavailable | 3 consecutive command failures abort with `probe_unavailable` and `healthUnknown=true`; the receipt does not claim hardware damage. |
| L4 external occupancy after the lease | 2 consecutive `proc-mem` observations containing processes without this run's token abort L4 with `external_occupancy` (category `occupancy`). This is the loss-of-exclusivity signal; L3 legal sharing is recorded as `shared(N)` and never aborts. |
| L4 occupancy not observable | 3 consecutive unreadable `proc-mem` probes abort with `occupancy_unknown` (category `occupancy`) because exclusivity cannot be confirmed. |
| Liveness | There is no CPU-time, log-silence, or progress-file stall heuristic. A silent long NPU kernel on a healthy card is indistinguishable from a hang, so it is left to the existing recipe deadline; this layer never invents an earlier kill. |
| Recipe deadline with an escaped pipe holder | `run_host_bash` enforces the recipe timeout, but a descendant that escaped its process group can keep the stdout/stderr pipes open and delay its final `communicate()` beyond the deadline. Once the deadline plus a 0.25 s grace passes, the adapter sweeps only processes carrying this run's token, which closes the pipes and lets the existing runner return its `TimeoutExpired`. This is deadline enforcement, not a stall heuristic; the sweep is recorded as `cleanup.deadlineSweep`. |
| Cleanup and lease ordering | On abort or completion, token processes get SIGTERM with a 2 s grace, then SIGKILL with a 10 s wait and periodic token rescans. The worker is joined (8 s bound) before `finally` releases the device window. |
| Unreclaimable device residue | D-state work cannot be guaranteed to die or be reaped in finite time. When the bounded sweep still finds this run's token processes, their `{pid, startTime, comm}` identities and the failure reason/phase/card are atomically written to `<device-lease>.quarantine.json` (mode 0600) before the flock is released. A successor takes the same per-card lock, then checks the marker before occupancy or workload: any live or indeterminate identity returns `BLOCKED_ENV` (75) with `occupancy="quarantined"`. Once every recorded identity is confirmed exited, the marker is removed and the card becomes usable again. Pid reuse is rejected by `startTime`. No global disable is used and other tenants are never touched. |

A marker with no process identity, invalid JSON, an unexpected schema, or an
unreadable file is treated as indeterminate and stays blocking rather than
guessing that the card is clear.

Host-simulated acceptance (no live card, no real fault injection) measured a
simulated fatal health fault from first bad probe to process exit/lease release
in 4.79 s, L4 external occupancy in 4.82 s, a probe self-hang in 1.61 s, a
disappeared card in 1.75 s, a 1.0 s recipe deadline with an escaped
pipe-holding descendant in 1.74 s, and an unreapable escaped descendant
returning a bounded quarantined result in 0.82 s. These runs shortened some
probe intervals and probe timeouts;
every phase remains independently bounded by the 5 s per-probe timeout, 2 s
SIGTERM grace, 10 s SIGKILL wait, and 8 s worker join above, so a stuck probe or
reap cannot consume the 1200 s L3 or 3300 s L4 budget.

The observation's `healthMonitoring` block records `health`, `healthUnknown`,
`occupancy`, own/external/unknown PIDs, `probeFailures`, `fatalHealthStreak`,
`nonOkHealthStreak`, `missingMappingStreak`, `stuckProbePids`,
`stuckProbePidsObserved`, the last probe
and mapping check, `unknownSignals`, `abort` (kind, reason, phase, device, NPU
ID, occupancy, stuck probe PIDs), and `cleanup` (PIDs, SIGTERM/SIGKILL,
`survivingProcesses`, `quarantine`, `probeCleanup`, `deadlineSweep`,
`incomplete`, `workerJoined`). L4 internal fault
counters are not exposed by the host SMI interface, so every L4 snapshot lists
`l4_internal_fault_counters_unavailable` in `unknownSignals` instead of
inferring a card state. Aborts preserve the phase that was active and any
partial L3 evidence/JUnit/workload facts already written; an aborted run never
promotes and never reports a fabricated hardware diagnosis.



```bash
agent-gates doctor --adapter tilefoundry-npu --source /abs/path/to/TileFoundry --device-id 1 --json
agent-gates doctor --adapter tilefoundry-npu --source /abs/path/to/TileFoundry --device-id 1 --gate tilefoundry-npu-l3-promotion --target level4/attention_residual --candidate examples/ascend950pr_ops/attention_residual/kernels.py --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-device --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-l3-promotion --target level4/attention_residual --candidate examples/ascend950pr_ops/attention_residual/kernels.py --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-cannbench --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-cannbench-remote --json
agent-gates wait --execution-id <EXECUTION_ID> --timeout 660 --json
agent-gates verify-requirements --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-l3-promotion --run-id <RUN_ID> --target level4/attention_residual --candidate examples/ascend950pr_ops/attention_residual/kernels.py --json
agent-gates verify-requirements --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-cannbench --run-id <RUN_ID> --json
```
