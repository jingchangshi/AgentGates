# Machine configuration template for the host-local TileLang Adapter.
#
# Copy this file to env.sh beside it and fill in this machine. env.sh is
# git-ignored; this template is committed and every value stays empty so no
# other machine path ships in the package.
#
# The file is data, never code: strict NAME=value lines only, no shell
# expansion, no comments on a value line, no duplicate keys.

# CANN toolkit initialization script, sourced before the conda environment.
TILELANG_NPU_CANN_SETUP=

# The conda shell hook (conda.sh) and the environment holding torch, torch_npu
# and the TileLang runtime dependencies.
TILELANG_NPU_CONDA_SH=
TILELANG_NPU_CONDA_ENV=
TILELANG_NPU_CONDA_ENVS_PATH=

# The interpreter the build is configured against and the examples run under.
# It must be the interpreter inside TILELANG_NPU_CONDA_ENV: the CMake configure
# step records it, and a mismatch produces extension modules for another Python.
TILELANG_NPU_PYTHON=

# Optional: the CMake binary that regenerates the build tree. Leave empty to use
# the cmake the activated environment provides. Name it explicitly when a tree
# already exists and the environment ships a different CMake major version, so
# the operator decides which generator rewrites that tree.
TILELANG_NPU_CMAKE_BIN=

# The build directory, relative to the checkout. The engine leases the same
# directory this Adapter writes.
TILELANG_NPU_BUILD_SUBDIR=

# Ninja job count. Size it to the memory this host actually gives the build, not
# to the core count: a translation unit in this project peaks near 800 MiB, so a
# job count sized to cores can exhaust a small memory cgroup and turn the build
# into swap thrash.
TILELANG_NPU_NINJA_JOBS=

# Optional: the fixed example tests the regression runs, comma separated and
# relative to the checkout. Leave empty for the reviewed default list.
TILELANG_NPU_EXAMPLE_TESTS=
