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
# the launching shell has no effect. The one exception is BSKA_NPU_DEVICE_ID,
# which authorizes one run and is passed as --device-id or an export.
#
# Only plain NAME=value lines are read. '#' comments and blank lines are
# allowed. The file is parsed as data and never executed. Quotes around a value
# are optional and stripped.
#
# There is no container: the Gate runs as the invoking user, on the host. That
# is deliberate. A container's login shell pinned ASCEND_RT_VISIBLE_DEVICES and
# its root identity resolved a different Torch than the one the native extension
# was compiled against, so a container run could pass every check and still
# score nothing.
#
# ---------------------------------------------------------------------------
# Machine values
# ---------------------------------------------------------------------------
# CANN set_env.sh, sourced before anything imports torch_npu. It must be
# complete enough for Ascend950PR: libhccl.so must resolve and ASCEND_OPP_PATH
# must carry the 950 operator package.
BSKA_NPU_CANN_SETUP=

# The conda environment holding torch, torch_npu, triton and helix. The Gate
# uses <CONDA_ENVS_PATH>/<CONDA_ENV>/bin/python directly and needs no
# activation. This interpreter must be the one the native extension was
# compiled against; see BSKA_NPU_CANNBENCH_REPO below.
BSKA_NPU_CONDA_ENV=
BSKA_NPU_CONDA_ENVS_PATH=

# The CANNBench checkout: the directory holding tools/run_cannbench.py and
# operators/. The native extension is expected at
# <CANNBENCH_REPO>/.cannbench-build/cann_bench_utils.
BSKA_NPU_CANNBENCH_REPO=

# The operator the --cannbench step scores, as levelN/name.
BSKA_NPU_CANNBENCH_TASK=

# Optional: the build target for the native extension. Leave empty for
# ascend950. Other values are ascend910b and ascend910_93.
# BSKA_NPU_CANNBENCH_ARCH=ascend950

# Build selection accepted by the engine so it can take a resource lease.
# BSKA_NPU_BUILD_SUBDIR=build

# Optional absolute paths of node and claude. Leave empty to use the one on
# PATH; the Gate then reports which it found.
BSKA_NPU_NODE_BIN=
BSKA_NPU_CLAUDE_BIN=

# ---------------------------------------------------------------------------
# Exact triton_ascend build - read by the Gate
# ---------------------------------------------------------------------------
# Vendor triton_ascend wheels change the TLA API between daily drops, so naming
# one exact build is what makes an optimize run reproducible. This is the
# distribution version string, i.e. importlib.metadata.version("triton-ascend").
# When empty, the Gate reports the installed version but does not pass: the
# Adapter has no way to prove the build is the reviewed one.
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
