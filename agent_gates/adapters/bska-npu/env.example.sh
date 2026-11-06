# CANN, Conda, container identity and the BiShengKernelAgent checkout - the
# bska-npu Adapter.
#
# This is the single machine file for the bska-npu Adapter. Two readers use it:
#
#   * scripts/bska_gate.py   - --check-config and --smoke
#   * the AgentGates engine  - to resolve the build directory it leases
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
# the launching shell has no effect. The one exception is BSKA_NPU_DEVICE_ID,
# which authorizes one run and is passed as --device-id or an export.
#
# Only plain NAME=value lines are read. '#' comments and blank lines are
# allowed. The file is parsed as data and never executed. Quotes around a value
# are optional and stripped. All paths are CONTAINER paths; because /home and
# /data are bind-mounted at the same path, host and container paths are equal.
#
# ---------------------------------------------------------------------------
# Machine values
# ---------------------------------------------------------------------------
# CANN set_env.sh. It must be complete enough for torch_npu: it must provide
# libhccl.so and the Ascend950PR operator package under ASCEND_OPP_PATH. The
# /home/<user>/CANN/cann tree on this host is NOT complete (no libhccl.so, no
# 950 ops), so this points at the complete 9.3.0 toolkit under /data.
BSKA_NPU_CANN_SETUP=

# Optional extra directory prepended to LD_LIBRARY_PATH after CANN is sourced.
BSKA_NPU_CANN_EXTRA_LIB=

# Conda: conda.sh (optional) and the environment that provides torch, torch_npu,
# triton and helix. CONDA_ENVS_PATH holds the environment; the Gate uses
# <CONDA_ENVS_PATH>/<CONDA_ENV>/bin/python directly and does not need a conda
# activation.
BSKA_NPU_CONDA_SH=
BSKA_NPU_CONDA_ENV=
BSKA_NPU_CONDA_ENVS_PATH=
BSKA_NPU_PYTHON_VERSION=3.11

# The BiShengKernelAgent checkout (host and container path are identical under
# the /home bind mount). Informational for the Gate's reports.
BSKA_NPU_HELIX_REPO=

# Build selection accepted by the engine so it can take a resource lease.
# BSKA_NPU_BUILD_SUBDIR=build

# Optional absolute paths of node and claude inside the container.
BSKA_NPU_NODE_BIN=/usr/local/bin/node
BSKA_NPU_CLAUDE_BIN=/usr/local/bin/claude

# ---------------------------------------------------------------------------
# Container identity - read by the Gate
# ---------------------------------------------------------------------------
BSKA_NPU_CONTAINER_NAME=
BSKA_NPU_CONTAINER_USER=
BSKA_NPU_CONTAINER_HOME=

# ---------------------------------------------------------------------------
# Illustrative shapes ONLY (not values from any machine)
# ---------------------------------------------------------------------------
# BSKA_NPU_CANN_SETUP=/data/mcl/Ascend/9.3.0.w/ascend-toolkit/set_env.sh
# BSKA_NPU_CONDA_SH=/data/miniconda3/etc/profile.d/conda.sh
# BSKA_NPU_CONDA_ENV=bska-npu
# BSKA_NPU_CONDA_ENVS_PATH=/home/<user>/.conda/envs
# BSKA_NPU_HELIX_REPO=/home/<user>/workspace/BiShengKernelAgent
# BSKA_NPU_CONTAINER_NAME=<user>-bska
# BSKA_NPU_CONTAINER_USER=<user>
# BSKA_NPU_CONTAINER_HOME=/home/<user>
