# Machine configuration template for the host-local TileFoundry Ascend NPU Adapter.
#
# Copy to `env.sh` beside this template (git-ignored) and fill in host-specific paths:
#
#     cd <repo-root>/agent_gates/adapters/tilefoundry-npu
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# Lookup order:
#   1. $TILEFOUNDRY_NPU_ENV_FILE (explicit absolute path; must exist)
#   2. agent_gates/adapters/tilefoundry-npu/env.sh beside this template
#
# Format & strictness rules:
#   * Plain `NAME=VALUE` data lines only (no shell execution, no `$()` or backtick
#     expansion, no comments after values, no duplicate keys).
#   * Conflicting `TILEFOUNDRY_NPU_*` exports in the calling shell are refused by
#     name so stale shell state cannot mask this file.
#
# Launch-only parameters (NEVER write these in env.sh):
#   * TILEFOUNDRY_NPU_DEVICE_ID: passed per run via `--device-id` or local config.toml.
#   * TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE: forwarded by AgentGates devices config.

# ---------------------------------------------------------------------------
# Toolchain — host-local CANN & Conda
# ---------------------------------------------------------------------------
# Required: absolute host path to CANN `set_env.sh` (e.g. <cann-root>/cann-9.3.0/set_env.sh).
# Sourced ONLY in `tilefoundry-npu-device`. The CPU gate (`tilefoundry-npu-cpu`)
# intentionally skips CANN initialization and sets TORCH_DEVICE_BACKEND_AUTOLOAD=0
# to avoid native library collisions (such as ortools vs CANN).
TILEFOUNDRY_NPU_CANN_SETUP=

# Required: absolute host path to Conda shell hook (e.g. <conda-root>/etc/profile.d/conda.sh).
TILEFOUNDRY_NPU_CONDA_SH=

# Required: Conda environment name (e.g. tilefoundry-npu) containing matched
# torch (+cpu), torch_npu, pytest, z3, and TileFoundry runtime dependencies.
TILEFOUNDRY_NPU_CONDA_ENV=

# Optional: absolute host path to custom Conda envs directory (e.g. <user-home>/.conda/envs)
# when TILEFOUNDRY_NPU_CONDA_ENV lives outside `<conda-root>/envs`.
TILEFOUNDRY_NPU_CONDA_ENVS_PATH=

# Optional: explicit absolute host path to python3 inside TILEFOUNDRY_NPU_CONDA_ENV.
# Leave empty to derive `<TILEFOUNDRY_NPU_CONDA_ENVS_PATH>/<env>/bin/python3`
# (or `<conda-root>/envs/<env>/bin/python3` from TILEFOUNDRY_NPU_CONDA_SH).
TILEFOUNDRY_NPU_PYTHON=

# ---------------------------------------------------------------------------
# Reused TileLang checkout build (built by `tilelang-npu`, never rebuilt here)
# ---------------------------------------------------------------------------
# Optional: absolute host paths to the shared TileLang checkout and its `build`
# directory compiled by `tilelang-npu` (`tilelang-npu-build` produces cp39-abi3
# `.abi3.so` extensions and `.agent_gates_build_stamp.json`).
# When set:
#   * `TILEFOUNDRY_NPU_TILELANG_SOURCE` is prepended to `PYTHONPATH` ahead of
#     `site-packages` so TileFoundry runs against the live built checkout.
#   * `TILEFOUNDRY_NPU_TILELANG_BUILD` must resolve to `<TILELANG_SOURCE>/build`
#     (leave empty to default to `<TILELANG_SOURCE>/build`).
#   * `--check-config` / `doctor` and all gates verify `libtilelang.so`, all four
#     `*.abi3.so` extensions, and `.agent_gates_build_stamp.json`; if upstream
#     native/Cython sources changed after build, this adapter fails closed with
#     `BLOCKED_ENV` (exit 75) instead of rebuilding TileLang itself.
TILEFOUNDRY_NPU_TILELANG_SOURCE=
TILEFOUNDRY_NPU_TILELANG_BUILD=

# ---------------------------------------------------------------------------
# Test selection — CPU and NPU pytest targets
# ---------------------------------------------------------------------------
# Optional: comma- or whitespace-separated list of RELATIVE pytest file or directory
# paths inside the TileFoundry checkout (`--source`), e.g.:
#   TILEFOUNDRY_NPU_CPU_TESTS=tests/tilefoundry,tests/ascend/passes/test_foo.py
# Rules:
#   * Leading `/` and `..` path traversal are rejected.
#   * `--check-config` / `doctor` verifies every configured relative path exists.
#   * Leave empty to run the 5 default CPU test directories without CANN sourced:
#     tests/tilefoundry,tests/ascend/target,tests/ascend/ir,tests/ascend/passes,tests/ascend/operators
TILEFOUNDRY_NPU_CPU_TESTS=

# Optional: comma- or whitespace-separated list of RELATIVE pytest file or directory
# paths inside the TileFoundry checkout (`--source`), e.g.:
#   TILEFOUNDRY_NPU_DEVICE_TESTS=tests/ascend/e2e/test_vadd_npu.py
# Rules:
#   * Leading `/` and `..` path traversal are rejected.
#   * `--check-config` / `doctor` verifies every listed path exists when non-empty.
#   * Leave empty by default: `tilefoundry-npu-device` always executes its built-in
#     NPU runtime and numerical verification probe (`import tilefoundry`, FP32 vector
#     add, BF16 Cube `matmul`, device memory > 0, `torch.npu.synchronize()`) inside the
#     `npu:<id>` lease window. Only when non-empty does it additionally run `pytest` on
#     these relative NPU test paths inside the same lease window and require >=1 passed
#     test with 0 failures.
TILEFOUNDRY_NPU_DEVICE_TESTS=

# ---------------------------------------------------------------------------
# Build / lease directory selection
# ---------------------------------------------------------------------------
# Single relative directory name under `--source` used by AgentGates for its
# `build:<dir>` resource lease. Keep `build` (or comment the line out to use the
# default `build`; do not set an empty `TILEFOUNDRY_NPU_BUILD_SUBDIR=` value).
TILEFOUNDRY_NPU_BUILD_SUBDIR=build

# ---------------------------------------------------------------------------
# CANNBench operator accuracy & performance gates (optional; required only for
# `tilefoundry-npu-cannbench-prep`, `tilefoundry-npu-cannbench`, and
# `tilefoundry-npu-cannbench-remote`)
# ---------------------------------------------------------------------------
# Required for CANNBench gates: absolute host path to the official CANNBench
# checkout (e.g. `<user-home>/workspace/cann-bench`) containing
# `scripts/run_evaluation.sh`, `src/kernel_eval`,
# `src/cann_bench_utils`, and `tasks/metadata`.
TILEFOUNDRY_NPU_CANNBENCH_REPO=

# Required for CANNBench gates: isolated host work root (e.g.
# `<user-home>/workspace/tilefoundry-cannbench-work`) placed OUTSIDE both
# `--source` and `TILEFOUNDRY_NPU_CANNBENCH_REPO`. Stores the ABI-isolated
# `cann_bench_utils` build (`<work-root>/cannbench-utils/...`), persisted
# CANNBench reports (`<work-root>/reports/...`), and remote submission
# archives/receipts (`<work-root>/submissions/...`).
TILEFOUNDRY_NPU_WORK_ROOT=

# Optional for local CANNBench gates: NPU build arch for `cann_bench_utils`
# (`ascend950`, `ascend910b`, or `ascend910_93`). Leave empty to use `ascend950`.
TILEFOUNDRY_NPU_CANNBENCH_ARCH=

# Optional for CANNBench gates: relative path inside `--source` (no leading `/`
# or `..`) pointing to the JSON target declaration file. Leave empty to default to:
#   examples/ascend950pr_ops/cannbench_targets.json
# Expected JSON schema:
#   {"targets": [{"task": "level4/attention_residual", "candidate": "examples/ascend950pr_ops/attention_residual/kernels.py"}]}
TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE=

# Optional fallback for CANNBench gates when the target JSON file in `--source`
# does not exist: comma/whitespace-separated `levelN/op:relative/candidate.py` pairs, e.g.:
#   TILEFOUNDRY_NPU_CANNBENCH_TARGETS=level4/attention_residual:examples/ascend950pr_ops/attention_residual/kernels.py
TILEFOUNDRY_NPU_CANNBENCH_TARGETS=

# ---------------------------------------------------------------------------
# Remote CANNBench submission via BenchSite MCP (https://cannbench.com/docs/mcp)
# Required only for `tilefoundry-npu-cannbench-remote`
# ---------------------------------------------------------------------------
# Optional: BenchSite MCP Streamable HTTP endpoint URL (defaults to https://cannbench.com/mcp).
TILEFOUNDRY_NPU_CANNBENCH_MCP_URL=

# Required for remote submit (inline token OR token file): Bearer token for BenchSite MCP.
TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN=

# Alternative to inline token: absolute host path to a chmod-600 file containing the Bearer token.
TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE=

# Optional: explicit `benchmark_version` passed to `submit_kernel` (e.g. `1.1.2` or `2.1.0`).
TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION=

# Optional: explicit `requested_pool_id` passed to `submit_kernel` (auto-routed when empty).
TILEFOUNDRY_NPU_CANNBENCH_POOL_ID=

# Optional: `aggregation_token` for `list_runner_pools` authorization queries.
TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN=

# Optional: explicit user consent version for independent execution environments (e.g. `v1`).
# Leave empty unless the user has explicitly consented (pool/team membership is NOT consent).
TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT=

# ---------------------------------------------------------------------------
# Optional neuro-symbolic agent LLM & reference repo configuration
# ---------------------------------------------------------------------------
# Optional backend: agy_cli, openai_compat, claude_cli, codex_cli, custom_cli, or mock.
TILEFOUNDRY_NPU_LLM_BACKEND=

# HTTP endpoint base URL, for example https://llm.example.invalid/v1.
TILEFOUNDRY_NPU_LLM_API_BASE=

# Name of the process environment variable holding the API credential.
# Use a name such as MODEL_PROVIDER_API_KEY. Keep its value outside env.sh.
TILEFOUNDRY_NPU_LLM_API_KEY_ENV=

# Optional model IDs for the four generation roles.
TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL=
TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL=
TILEFOUNDRY_NPU_LLM_ESCALATE_MODEL=
TILEFOUNDRY_NPU_LLM_REPAIR_MODEL=

# Optional effort cap. MAX_EFFORT takes precedence over EFFORT_CAP.
TILEFOUNDRY_NPU_LLM_MAX_EFFORT=
TILEFOUNDRY_NPU_LLM_EFFORT_CAP=

# Set to 1 to send reasoning_effort to a compatible HTTP endpoint.
TILEFOUNDRY_NPU_LLM_PASS_REASONING_EFFORT=

# Set to 1 to stream OpenAI-compatible chat completions using server-sent events.
TILEFOUNDRY_NPU_LLM_STREAM=

# Optional concurrency cap (positive integer) and role time budgets (positive seconds).
TILEFOUNDRY_NPU_LLM_MAX_CONCURRENCY=
TILEFOUNDRY_NPU_LLM_EXPLOIT_TIMEOUT_SEC=
TILEFOUNDRY_NPU_LLM_REASONING_TIMEOUT_SEC=

# Optional CLI executable and quoted command template for custom_cli.
TILEFOUNDRY_NPU_LLM_CLI_BIN=
TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE=

# Optional absolute host path to the `agy` CLI binary (e.g. `<user-home>/.local/bin/agy`).
TILEFOUNDRY_NPU_AGY_BIN=

# Optional legacy model IDs for exploit, explore, and escalation with agy_cli.
TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL=
TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL=
TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL=

# Optional common CLI flags forwarded to headless `agy` worker subprocesses.
# Quote the full string when passing multiple space-separated flags, e.g. "--print --mode plan".
TILEFOUNDRY_NPU_AGY_COMMON_ARGS=

# Optional positive integer concurrency cap for parallel `agy` worker subprocesses.
TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY=

# Optional absolute host paths to local read-only Ascend 950 reference repositories
# used for offline inspection and Two-Snapshot IR Diff (never passed to runtime `agy`).
TILEFOUNDRY_NPU_TILELANG_REPO=
TILEFOUNDRY_NPU_CANNBOT_DSL_REPO=
TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO=
TILEFOUNDRY_NPU_ASC_DEVKIT_REPO=
