# CANN, Conda and the CANNBench checkout - the bska-npu Adapter.
#
# This is the single machine file for the bska-npu Adapter. Readers: the Gate
# (--check-config, --smoke, --cannbench) and the AgentGates engine, which
# resolves the build directory it leases from the same file.
#
# Create it:
#     cd <checkout>/agent_gates/adapters/bska-npu
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# env.sh is git-ignored: it holds machine paths and is never committed.
#
# Two locations, in this order:
#     1. BSKA_NPU_ENV_FILE, an explicit absolute path. It must exist.
#     2. adapters/bska-npu/env.sh, beside this template.
# There is deliberately no home-directory search.
#
# This file is the only source of these values. Exporting a BSKA_NPU_* key in
# the launching shell has no effect (and conflicting exports are refused).
# Launch-only parameters (NEVER write these in env.sh):
#   * BSKA_NPU_DEVICE_ID: passed per run via `--device-id` or local config.toml.
#   * BSKA_NPU_REQUIRE_EXCLUSIVE_DEVICE: forwarded by AgentGates devices config.
#
# Only plain NAME=value lines are read. '#' comments and blank lines are
# allowed. The file is parsed as data and never executed. Quotes around a value
# are optional and stripped.
#
# Host-local execution only: the Gate runs directly on the host as the invoking
# user, inside the target Conda Python environment.
#
# ---------------------------------------------------------------------------
# Machine values & task selection
# ---------------------------------------------------------------------------
# Required: CANN set_env.sh, sourced before anything imports torch_npu. It must
# be complete enough for Ascend950PR (libhccl.so + 950 OPP).
BSKA_NPU_CANN_SETUP=

# Required: Conda environment name and envs root holding torch, torch_npu,
# triton-ascend, and helix. The Gate executes
# `<BSKA_NPU_CONDA_ENVS_PATH>/<BSKA_NPU_CONDA_ENV>/bin/python` directly.
BSKA_NPU_CONDA_ENV=
BSKA_NPU_CONDA_ENVS_PATH=

# Required: absolute host path to the CANNBench checkout holding
# `tools/run_cannbench.py`, `operators/`, and `.cannbench-build/cann_bench_utils`.
BSKA_NPU_CANNBENCH_REPO=

# Required: operator task ID scored by `bska-npu-cannbench` in `levelN/name` form
# (e.g. `level1/swi_glu`), resolved under `<BSKA_NPU_CANNBENCH_REPO>/operators/<task>`.
BSKA_NPU_CANNBENCH_TASK=

# Optional: build target arch for the native extension. Leave empty/commented
# for `ascend950`; other valid values are `ascend910b` and `ascend910_93`.
# BSKA_NPU_CANNBENCH_ARCH=ascend950

# Optional: single relative directory name under `--source` used by AgentGates
# for its `build:<dir>` resource lease (commented = default `build`; do not set empty).
# BSKA_NPU_BUILD_SUBDIR=build

# Optional: absolute paths to `node` and `claude` binaries. Leave empty to resolve
# them from PATH.
BSKA_NPU_NODE_BIN=
BSKA_NPU_CLAUDE_BIN=

# ---------------------------------------------------------------------------
# Exact triton_ascend build - read by the Gate
# ---------------------------------------------------------------------------
# Required: exact distribution version string returned by
# `importlib.metadata.version("triton-ascend")` (e.g. `3.6.0.post...`). Leaving
# this empty or mismatched fails `--check-config` / `doctor`, so a run cannot
# pass against an unintended triton-ascend build.
BSKA_NPU_TRITON_ASCEND_VERSION=

# ---------------------------------------------------------------------------
# Illustrative shapes ONLY (not values from any machine)
# ---------------------------------------------------------------------------
# BSKA_NPU_CANN_SETUP=/data/<toolkit>/cann/set_env.sh
# BSKA_NPU_CONDA_ENV=bska-npu
# BSKA_NPU_CONDA_ENVS_PATH=/home/<user>/.conda/envs
# BSKA_NPU_CANNBENCH_REPO=/home/<user>/workspace/cannbot-ops
# BSKA_NPU_CANNBENCH_TASK=level1/swi_glu
# BSKA_NPU_TRITON_ASCEND_VERSION=<version>
# BSKA_NPU_NODE_BIN=/home/<user>/Softwares/node/bin/node
# BSKA_NPU_CLAUDE_BIN=/home/<user>/.local/bin/claude
