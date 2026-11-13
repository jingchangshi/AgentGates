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
