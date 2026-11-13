# CANN, Conda and the TLA test location — the host-local tla-npu Adapter.
#
# Host-first mode notice:
#   Legacy container execution (`docker/create-container.sh`) is QUARANTINED
#   (`DOCKER_MODE_DISABLED`). `scripts/tla_gate.py` now runs directly on the host
#   against host CANN + Conda. All paths below are HOST paths.
#
# Single configuration reader:
#   * scripts/tla_gate.py  — host config check & NPU E2E device Gate
#
# ---------------------------------------------------------------------------
# Create the file
# ---------------------------------------------------------------------------
# On a host with CANN and Conda, copy this template beside itself:
#
#     cd <checkout>/agent_gates/adapters/tla-npu
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# `env.sh` is git-ignored, so it is never committed and never leaves the machine.
#
# ---------------------------------------------------------------------------
# Two locations, in this order
# ---------------------------------------------------------------------------
#     1. $TLA_NPU_ENV_FILE, an explicit absolute path. It must exist.
#     2. agent_gates/adapters/tla-npu/env.sh, beside this template.
#
# ---------------------------------------------------------------------------
# This file is the only source of these values
# ---------------------------------------------------------------------------
# Exporting any `TLA_NPU_*` machine key in the launching shell that conflicts with
# this file is refused: the Gate reads machine facts from this file only.
#
# Launch-only parameters (NEVER write these in env.sh):
#   * TLA_NPU_DEVICE_ID: passed per run via `--device-id` or local config.toml.
#   * TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE: forwarded by AgentGates devices config.
#
# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
# Plain `NAME=value` data lines only (`#` comments and blank lines allowed). The
# Gate parses the file as data and never executes shell expansion. Quotes around a
# value are optional and stripped.
#
# All paths are HOST paths reachable directly by the invoking user.

# ---------------------------------------------------------------------------
# Host toolchain — read by the Gate
# ---------------------------------------------------------------------------
# CANN set_env.sh (absolute host path). Optional when CANN is absent on a dry host;
# required for real NPU execution and must define ASCEND_HOME_PATH. Loaded first
# so `import torch_npu` resolves libhccl.so.
TLA_NPU_CANN_SETUP=

# Conda shell hook (`conda.sh`) and target environment providing pytest, torch,
# torch_npu, and the vendor `triton_ascend` wheel. Set both or neither; setting
# only one is refused so the Gate cannot run against an unintended Python.
TLA_NPU_CONDA_SH=
TLA_NPU_CONDA_ENV=

# Where Conda searches for environments (e.g. `<user-home>/.conda/envs`), required
# whenever the environment lives outside the Conda installation prefix.
TLA_NPU_CONDA_ENVS_PATH=

# ---------------------------------------------------------------------------
# Legacy container identity — accepted for compatibility, IGNORED on host route
# ---------------------------------------------------------------------------
# Docker mode is quarantined; leave all three empty on host-local machines. They
# remain in the allowed key set only so existing files continue to parse.
TLA_NPU_CONTAINER_NAME=
TLA_NPU_CONTAINER_USER=
TLA_NPU_CONTAINER_HOME=
# TLA_NPU_CONTAINER_SOURCE=/workspace/triton-ascend

# ---------------------------------------------------------------------------
# Build selection — lease directory name (accepted by parser & engine)
# ---------------------------------------------------------------------------
# `tla-npu` compiles kernels via prebuilt wheels and writes no source build tree,
# so it declares no Ninja job count. Leave the line commented out to use `build`
# for the engine's resource lease; do not set an empty `TLA_NPU_BUILD_SUBDIR=`.
# TLA_NPU_BUILD_SUBDIR=build

# ---------------------------------------------------------------------------
# Test, cache & provisioning paths
# ---------------------------------------------------------------------------
# Required by setup/host test run: directory holding vendor wheels (used only by
# manual provisioning scripts, ignored by the runtime Gate).
TLA_NPU_WHEEL_DIR=

# Required by the Gate: absolute host directory holding `test_tla.py`.
# This MUST be separate from `--source` (the `triton-ascend` checkout under test)
# and separate from `TLA_NPU_CACHE_DIR`. Neither `--source` nor `TLA_NPU_TEST_DIR`
# is written to by the Gate.
TLA_NPU_TEST_DIR=

# Required by the Gate: dedicated parent directory for TRITON_CACHE_DIR.
# Must lie OUTSIDE both `--source` and `TLA_NPU_TEST_DIR`. At runtime the Gate
# creates a fresh temporary subdirectory underneath `TLA_NPU_CACHE_DIR` per run so
# concurrent or stale runs never pollute kernel compilation evidence.
TLA_NPU_CACHE_DIR=

# Optional provisioning settings (ignored by the runtime Gate).
# TLA_NPU_PIP_INDEX=
TLA_NPU_PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Illustrative shapes ONLY (host paths)
# ---------------------------------------------------------------------------
# TLA_NPU_CANN_SETUP=/opt/cann/cann-9.x/set_env.sh
# TLA_NPU_CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
# TLA_NPU_CONDA_ENV=tla-env
# TLA_NPU_CONDA_ENVS_PATH=/home/u/.conda/envs
# TLA_NPU_TEST_DIR=/home/u/workspace/agent_tla
# TLA_NPU_CACHE_DIR=/home/u/.cache/tla-npu-gate
