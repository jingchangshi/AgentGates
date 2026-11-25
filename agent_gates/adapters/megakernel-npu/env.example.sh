# Machine configuration template for the host-local MegaKernel Ascend NPU Adapter.
#
# Copy to `env.sh` beside this template (git-ignored) and fill in host paths:
#
#     cd <repo-root>/agent_gates/adapters/megakernel-npu
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# Lookup order:
#   1. $MEGAKERNEL_NPU_ENV_FILE (explicit absolute path; must exist)
#   2. agent_gates/adapters/megakernel-npu/env.sh beside this template
#
# Strict data rules:
#   * Plain `NAME=VALUE` lines only (no shell expansion or command substitution).
#   * Conflicting `MEGAKERNEL_NPU_*` exports in the calling shell are refused.
#
# Launch-only parameters (NEVER write these in env.sh):
#   * MEGAKERNEL_NPU_DEVICE_ID: passed per run via `--device-id` or config.toml.
#   * MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE: forwarded by AgentGates devices config.

# ---------------------------------------------------------------------------
# Toolchain — host-local CANN, Conda (Python 3.11 + torch==2.7.1), & LLVM
# ---------------------------------------------------------------------------
# Required: absolute host path to CANN `set_env.sh`.
MEGAKERNEL_NPU_CANN_SETUP=

# Required: absolute host path to Conda shell hook (`conda.sh`).
MEGAKERNEL_NPU_CONDA_SH=

# Required: Conda environment name (Python 3.11 + torch==2.7.1 + torch_npu==2.7.1.post4).
MEGAKERNEL_NPU_CONDA_ENV=

# Optional: absolute host path to custom Conda envs directory.
MEGAKERNEL_NPU_CONDA_ENVS_PATH=

# Optional: explicit absolute path to python3 inside MEGAKERNEL_NPU_CONDA_ENV.
MEGAKERNEL_NPU_PYTHON=

# Required for `--build`: absolute host path to prebuilt patched LLVM install dir
# (must contain `include/`, `lib/`, and `bin/FileCheck`).
MEGAKERNEL_NPU_LLVM_SYSPATH=

# Required for `--build`: isolated host work root OUTSIDE `--source`.
# Stores incremental staging tree (`stage/`), built wheels, build logs, and stamp.
MEGAKERNEL_NPU_WORK_ROOT=

# Optional: absolute host path to MegaKernel callback runtime directory
# (exported as `MEGA_KERNEL_HOME` during device runs).
MEGAKERNEL_NPU_RUNTIME_HOME=

# Optional: directory containing `bishengir-compile` if outside CANN `bin/`.
MEGAKERNEL_NPU_BISHENGIR_BIN=

# Optional: explicit absolute path to `cmake` (>=3.20,<4.0 recommended).
MEGAKERNEL_NPU_CMAKE_BIN=

# ---------------------------------------------------------------------------
# Build / lease settings
# ---------------------------------------------------------------------------
# Single relative directory name under `--source` used for `build:<dir>` lease.
MEGAKERNEL_NPU_BUILD_SUBDIR=build

# Build parallelism, strictly capped at <= 16 cores.
MEGAKERNEL_NPU_NINJA_JOBS=16

# Concurrent link jobs (`TRITON_PARALLEL_LINK_JOBS`, 1..4).
MEGAKERNEL_NPU_LINK_JOBS=2

# Optional: `ON` or `OFF` for `TRITON_ENABLE_ASCENDC_OPS` (default `OFF`).
MEGAKERNEL_NPU_ENABLE_ASCENDC_OPS=OFF

# Optional: `ON` or `OFF` to also build `3rdparty/shmem` wheel (default `OFF`).
MEGAKERNEL_NPU_BUILD_SHMEM=OFF

# ---------------------------------------------------------------------------
# Test selection — CPU and NPU device tests
# ---------------------------------------------------------------------------
# Optional: comma/whitespace-separated RELATIVE paths inside `--source` for CPU pytest.
# Leave empty to run:
#   python/triton_dist/mega_kernel_ascend/test/host,python/triton_dist/mega_kernel_ascend/test/ops/test_megakernel_dtype_contracts.py
MEGAKERNEL_NPU_CPU_TESTS=

# Optional: comma/whitespace-separated RELATIVE paths inside `--source` for extra
# device tests executed inside the `npu:<id>` lease window after the built-in probe.
MEGAKERNEL_NPU_DEVICE_TESTS=

# Optional: `auto` (default: pytest for directories, python3 for .py scripts),
# `script`, or `pytest`.
MEGAKERNEL_NPU_DEVICE_TEST_MODE=auto
