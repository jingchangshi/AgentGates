# CANN, Conda and the TLA test location — the tla-npu Adapter.
#
# This is the single configuration file for the TLA toolchain. Two readers use
# it and nothing else does:
#
#   * scripts/tla_gate.py  — the device Gate
#   * scripts/setup-python-env.sh — the one-time Python provisioning
#
# ---------------------------------------------------------------------------
# Create the file
# ---------------------------------------------------------------------------
# An absent file is not an error: it means "configure nothing". Both CANN and
# Conda are optional, so a Gate then runs with neither, which is correct on a
# machine without them and silently wrong on a machine with them. A run then
# fails deep inside the test with `ImportError` instead of at init.
# So on a machine that has CANN or Conda, create this file:
#
#     cd <checkout>/adapters/tla-npu
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# `env.sh` is git-ignored, so it is never committed and never leaves the
# machine. The Adapter manifest pins only `scripts/`, so the Gate reads this
# file at run time and no adapter install or upgrade overwrites it.
#
# ---------------------------------------------------------------------------
# Two locations, in this order
# ---------------------------------------------------------------------------
#     1. $TLA_NPU_ENV_FILE, an explicit absolute path. It must exist.
#     2. adapters/tla-npu/env.sh, beside this template.
#
# The second is found by walking up from the Gate to the checkout that owns it,
# so the same file serves a human and an agent working in this repository,
# whether the Gate runs from this checkout or from an installed Pack inside it.
#
# ---------------------------------------------------------------------------
# This file is the only source of these values
# ---------------------------------------------------------------------------
# Exporting any TLA_NPU_* key documented here in the launching shell has NO
# effect: the Gate reads them from this file only, so a stale export can never
# mask an edit.
#
# `TLA_NPU_DEVICE_ID` is NOT in this file. It authorizes one Gate run, not a
# machine, so it stays an export of the shell that launches the device Gate (or
# a CLI `--device-id`).
#
# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
# Only plain `NAME=value` lines are read. `#` comments and blank lines are
# allowed. The Gate parses the file as data and never executes it, so command
# substitution and variable expansion do not happen. Quotes around a value are
# optional and are stripped.
#
# The syntax is valid shell, so a human can source this file:
#
#     source <checkout>/adapters/tla-npu/env.sh
#
# All paths are CONTAINER paths, not host paths. They must be reachable through
# one of the container's bind mounts; with the usual `/home:/home` and
# `/data:/data` mounts the host and container path are identical.

# ---------------------------------------------------------------------------
# Toolchain — read by the Gate
# ---------------------------------------------------------------------------
# CANN set_env.sh. Optional; leave empty on a machine without CANN. When set,
# the file must be readable inside the container and must define
# ASCEND_HOME_PATH, or the Gate fails closed. CANN loads first, because
# `import torch_npu` resolves libhccl.so from the CANN layout.
TLA_NPU_CANN_SETUP=

# Conda: conda.sh and the environment that provides pytest, torch, torch_npu and
# the vendor triton_ascend wheel. Optional. Set both or neither; setting only one
# is an error, because the Gate must not silently run the test against a
# different interpreter.
#
# This names an installation that already exists. The provisioning script reuses
# it and never installs a second Conda while these are set.
TLA_NPU_CONDA_SH=
TLA_NPU_CONDA_ENV=

# Where Conda searches for environments. Required whenever the environment does
# not live under the installation prefix, which is the normal case for a user's
# own environments in `~/.conda/envs`. Without it `conda activate` fails with
# `EnvironmentNameNotFound` even though `conda env list` shows the environment.
TLA_NPU_CONDA_ENVS_PATH=

# ---------------------------------------------------------------------------
# Container identity — read by the Gate
# ---------------------------------------------------------------------------
# The container the Gate enters. It must be the container you created from
# `docker/config.env` in this directory, and these values must match that file's
# unprefixed `CONTAINER_NAME`, `CONTAINER_USER` and `CONTAINER_HOME`.
# The Gate cannot read `config.env` (it describes creation, not use), so this
# file is where the identity is stated once.
#
# The Gate refuses to run when any of these is empty: it will not guess a
# container, and it will not fall back to a shell export.
#
# `TLA_NPU_CONTAINER_USER` is accepted but not required: the device Gate runs as
# root, because the driver refuses a non-root opener once any namespace holds the
# card.
TLA_NPU_CONTAINER_NAME=
TLA_NPU_CONTAINER_USER=
# HOME inside the container. Normally covered by the /home bind mount.
TLA_NPU_CONTAINER_HOME=

# The container path of the directory holding the test, when the container sees
# it at a different path than the bind-mount mapping derives. Leave the line out
# to use the derived mapping, which is what a /home:/home mount needs.
#
# It confirms a path; it cannot invent one. The mapping is still computed from
# `docker inspect` and compared, so a value that disagrees with the mounts is
# refused.
#
# TLA_NPU_CONTAINER_SOURCE=/home/u/Packages/tla-20261030

# ---------------------------------------------------------------------------
# Build selection — accepted, unused
# ---------------------------------------------------------------------------
# This Gate writes no build tree. The engine still resolves a build directory
# name for every Adapter so it can take a resource lease, and it reads that name
# from this file; an empty value is a typo and is refused, so "unset" is spelled
# by leaving the line out.
#
# TLA_NPU_BUILD_SUBDIR=build

# ---------------------------------------------------------------------------
# Provisioning — read only by scripts/setup-python-env.sh
# ---------------------------------------------------------------------------
# The Gate accepts these keys and ignores them, so one file serves both readers.
# They are here rather than in the calling shell because a shell export leaves no
# record: it is invisible in review, lost when the shell exits, and a stale
# export silently provisions something other than what this file documents.

# Directory holding the vendor wheels: the triton_ascend wheel named in the task,
# and the torch/torch_npu pair it runs with. Required by the provisioning script.
TLA_NPU_WHEEL_DIR=

# The directory holding the Gate test file. This is NOT the source root passed to
# --source: the source root is the triton-ascend checkout whose TLA
# implementation is under test, and the test that exercises it lives separately.
# Both are inputs; neither is written to.
TLA_NPU_TEST_DIR=

# A dedicated TRITON_CACHE_DIR. The cache is written to, so it must be outside
# both the source root and the test directory, whose contents are hashed. The
# Gate refuses a cache path inside either tree rather than trusting a comment.
TLA_NPU_CACHE_DIR=

# Index for the small pure-Python dependencies. Leave the line out to use the
# documented public mirror; set it to an empty value to install from no index at
# all, which is what an offline run needs.
# TLA_NPU_PIP_INDEX=

# Interpreter version for a newly created environment. Ignored when the
# environment already exists.
TLA_NPU_PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Illustrative shapes ONLY
# ---------------------------------------------------------------------------
# The commented lines below show what a value looks like. They are not values
# from any machine and must not be copied literally.
#
# TLA_NPU_CONTAINER_NAME=tla-dev
# TLA_NPU_CONTAINER_USER=builder
# TLA_NPU_CONTAINER_HOME=/home/builder
# TLA_NPU_CANN_SETUP=/opt/cann/cann-9.x/set_env.sh
# TLA_NPU_CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
# TLA_NPU_CONDA_ENV=py311
# TLA_NPU_CONDA_ENVS_PATH=/home/u/.conda/envs
# TLA_NPU_WHEEL_DIR=/opt/wheels
# TLA_NPU_TEST_DIR=/home/u/workspace/agent_tla
# TLA_NPU_CACHE_DIR=/home/u/.cache/tla-npu-gate
