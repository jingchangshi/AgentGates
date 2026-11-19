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
| `tilefoundry-npu-cannbench-prep` | Verifies pure-Python CANNBench dependencies (`PyYAML`, `expecttest`, `ruamel.yaml`, `build`, `safetensors`), builds ABI-isolated `cann_bench_utils/_C.so` under `TILEFOUNDRY_NPU_WORK_ROOT`, and verifies `cann_bench_clone` on NPU | Sourced | Windowed (`deviceWindow`: 60 s) |
| `tilefoundry-npu-cannbench` | **Local testing mode**: depends on `tilefoundry-npu-cannbench-prep`; injects `--source` TileFoundry `PYTHONPATH`, runs official `cann-bench` `kernel_eval.staged_eval` against `tasks/<levelN>/<op>`, persists reports under `TILEFOUNDRY_NPU_WORK_ROOT/reports`, and records `score` / `avgSpeedup` | Sourced | Windowed (`deviceWindow`: 60 s) |
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
- `TILEFOUNDRY_NPU_DEVICE_TESTS`: optional comma- or whitespace-separated
  **relative** pytest paths under `--source` executed inside the NPU lease
  window after the built-in FP32/BF16 probe. Leave empty to run only the
  built-in runtime & compute probe.
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
- `TILEFOUNDRY_NPU_CANNBENCH_MCP_URL` / `TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN` /
  `TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE` / `TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION` /
  `TILEFOUNDRY_NPU_CANNBENCH_POOL_ID` / `TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN` /
  `TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT`:
  remote BenchSite MCP submission settings (`https://cannbench.com/docs/mcp`) used by
  `tilefoundry-npu-cannbench-remote`. `INDEPENDENT_ENV_CONSENT` (e.g. `v1`) is never
  set automatically and must only be supplied after explicit user consent.
- `TILEFOUNDRY_NPU_LLM_BACKEND` / `TILEFOUNDRY_NPU_AGY_BIN` /
  `TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL` / `TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL` /
  `TILEFOUNDRY_NPU_AGY_COMMON_ARGS` / `TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY`:
  optional headless `agy` LLM backend configuration forwarded when non-empty.
- `TILEFOUNDRY_NPU_TILELANG_REPO` / `TILEFOUNDRY_NPU_CANNBOT_DSL_REPO` /
  `TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO` / `TILEFOUNDRY_NPU_ASC_DEVKIT_REPO`:
  optional absolute host paths to read-only Ascend 950 reference checkouts forwarded when non-empty.

`TILEFOUNDRY_NPU_DEVICE_ID` and `TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE` are
per-run launch/config parameters supplied via `--device-id` or
`~/.config/agent-gates/config.toml`, never stored in `env.sh`.

## Verification commands

```bash
agent-gates doctor --adapter tilefoundry-npu --source /abs/path/to/TileFoundry --device-id 1 --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-device --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-cannbench --json
agent-gates submit --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-cannbench-remote --json
agent-gates wait --execution-id <EXECUTION_ID> --timeout 660 --json
agent-gates verify-requirements --workspace tilefoundry --adapter tilefoundry-npu --gate tilefoundry-npu-cannbench --run-id <RUN_ID> --json
```
