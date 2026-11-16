# Machine configuration template for the host-local TileLang Adapter (`tilelang-npu`).
#
# Copy this file to `env.sh` beside it and fill in this machine. `env.sh` is
# git-ignored; this template is committed and every value stays empty so no
# other machine path ships in the package.
#
# Lookup order:
#   1. $TILELANG_NPU_ENV_FILE (explicit absolute path; must exist)
#   2. agent_gates/adapters/tilelang-npu/env.sh beside this template
#
# The file is data, never code: strict `NAME=value` lines only, no shell
# expansion, no comments on a value line, no duplicate keys. Conflicting
# `TILELANG_NPU_*` shell exports are refused.
#
# Launch-only parameters (NEVER write these in env.sh):
#   * TILELANG_NPU_DEVICE_ID: passed per run via `--device-id` or local config.toml.
#   * TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE: forwarded by AgentGates devices config.

# Required: absolute host path to CANN toolkit initialization script, sourced
# before the conda environment (e.g. <cann-root>/cann/set_env.sh).
TILELANG_NPU_CANN_SETUP=

# Required: Conda shell hook (`<conda-root>/etc/profile.d/conda.sh`), target
# environment name holding torch/torch_npu/TileLang runtime deps, and optional
# envs directory (`<user-home>/.conda/envs` when outside `<conda-root>/envs`).
TILELANG_NPU_CONDA_SH=
TILELANG_NPU_CONDA_ENV=
TILELANG_NPU_CONDA_ENVS_PATH=

# Required: absolute path to the python3 interpreter inside TILELANG_NPU_CONDA_ENV.
# The CMake configure step records it, and `tilelang-npu-build` compiles Cython
# extensions with Python Stable ABI (`cp39-abi3`, `.abi3.so`) + records
# `.agent_gates_build_stamp.json` so downstream adapters (`forgekernel-ascend-npu`,
# `tilefoundry-npu`) across Python >= 3.10 can reuse `<checkout>/build` directly.
TILELANG_NPU_PYTHON=

# Optional: explicit absolute path to the CMake binary that regenerates the build
# tree. Leave empty to use the `cmake` provided by the activated environment.
# Name it explicitly when a build tree already exists and the environment ships
# a different CMake major version, so the operator decides which generator runs.
TILELANG_NPU_CMAKE_BIN=

# Optional in live `env.sh`: single relative directory name under the checkout
# used for the CMake/Ninja build tree and AgentGates `build:<dir>` resource lease.
# In this empty template all keys remain `KEY=`; in your live `env.sh`, either set
# e.g. `TILELANG_NPU_BUILD_SUBDIR=build` or comment the line out to use `build`
# (the parser refuses a present-but-empty value in live runs).
TILELANG_NPU_BUILD_SUBDIR=

# Optional in live `env.sh`: positive integer Ninja job count. Size it to the
# host RAM available (a translation unit peaks near 800 MiB, so sizing blindly to
# core count can trigger OOM/swap). In your live `env.sh`, either set e.g.
# `TILELANG_NPU_NINJA_JOBS=32` or comment the line out to use `$(nproc)`
# (do not leave an empty `TILELANG_NPU_NINJA_JOBS=` line in live `env.sh`).
TILELANG_NPU_NINJA_JOBS=

# Optional: comma-separated list of RELATIVE `.py` example test files inside the
# TileLang checkout (no leading `/`, no `..`, no globs). Leave empty to run the 3
# reviewed default Ascend example tests:
#   examples/ascend/test_vecadd.py,examples/ascend/test_rmsnorm.py,examples/ascend/test_per_token_cast_to_fp8.py
TILELANG_NPU_EXAMPLE_TESTS=
