# CANN, Python, the reused TileLang build and the private venv — the
# forgekernel-ascend-npu Adapter.
#
# This is the single configuration file for this Adapter. Two readers use it and
# nothing else does:
#
#   * scripts/fk_gate.py — the preflight, smoke and build Gates
#   * the engine        — to resolve the build directory it takes a lease on
#
# The smoke Gate needs only FK_ASCEND_PYTHON (and FK_ASCEND_CANN_SETUP when the
# backend needs it), so it is runnable before a ForgeKernel installation or a
# TileLang build exists. Everything below is for the build Gate.
#
# ---------------------------------------------------------------------------
# Create the file
# ---------------------------------------------------------------------------
# An absent file is not an error: it means "configure nothing", and the Gate then
# refuses with a message naming what is missing. On this machine, create it:
#
#     cd <checkout>/agent_gates/adapters/forgekernel-ascend-npu
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# `env.sh` is git-ignored, so it is never committed and never leaves the machine.
# The Adapter manifest pins only `scripts/`, so the Gate reads this file at run
# time and no adapter install or upgrade overwrites it.
#
# ---------------------------------------------------------------------------
# Two locations, in this order
# ---------------------------------------------------------------------------
#     1. $FK_ASCEND_ENV_FILE, an explicit absolute path. It must exist.
#     2. adapters/forgekernel-ascend-npu/env.sh, beside this template.
#
# There is deliberately no home-directory search: no Gate may read a machine
# file nobody reviewed.
#
# ---------------------------------------------------------------------------
# This file is the only source of these values
# ---------------------------------------------------------------------------
# Exporting any FK_ASCEND_* key documented here in the launching shell has NO
# effect: the Gate reads them from this file only, so a stale export can never
# mask an edit. An export that DISAGREES with the file is refused outright rather
# than silently overridden.
#
# `FK_ASCEND_DEVICE_ID` is NOT in this file. It authorizes one run, not a
# machine, so it stays an export of the shell that launches the Gate (or a CLI
# `--device-id`).
#
# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
# Only plain `NAME=value` lines are read. `#` comments and blank lines are
# allowed. The Gate parses the file as data and never executes it, so command
# substitution and variable expansion do not happen. Quotes around a value are
# optional and are stripped.
#
# All paths are HOST paths: this Adapter uses host-local delivery, not a
# container. There is no Docker assumption here.

# ---------------------------------------------------------------------------
# Toolchain — read by the Gate
# ---------------------------------------------------------------------------
# CANN set_env.sh. Required for any step that touches the device. It is sourced
# by the Gate before any device work, FIRST, because `import torch_npu` resolves
# libhccl.so from the CANN layout. The smoke Gate sources it too, so a run does
# not depend on the launching shell having been initialized.
FK_ASCEND_CANN_SETUP=

# The interpreter used to create the private environment. An absolute path to an
# existing executable; the Adapter refuses anything else.
FK_ASCEND_PYTHON=

# The REUSED TileLang checkout and its EXISTING build. This Adapter never
# configures or compiles TileLang: both keys name something that already exists.
# The build must contain libtilelang*.so or the Gate stops with a diagnosis
# instead of starting a multi-hour build nobody approved.
FK_ASCEND_TILELANG_SOURCE=
FK_ASCEND_TILELANG_BUILD=

# A dedicated TileLang kernel cache for this Adapter. Never the shared default:
# another run deleting or reusing that cache would change what this run measured.
# FK_ASCEND_TILELANG_CACHE_DIR=

# ---------------------------------------------------------------------------
# Work root and private environment — read by the Gate
# ---------------------------------------------------------------------------
# The runtime work root: runs, caches and artifacts live here, outside both the
# source checkout and the evidence directory.
#
# Required, not optional. The build installs from a verified COPY of the source
# (setuptools would otherwise write build/ into the checkout and move the source
# identity), and the copy is kept at <work-root>/build-src/<commit>/. A rebuild of
# the same commit reuses that path, so it is replaced rather than accumulated.
FK_ASCEND_WORK_ROOT=

# The private virtual environment ForgeKernel is installed into. Deleting it
# forces a clean reinstall; an existing one is reused, so remove it if the source
# it was built from has changed. It is created
# with --system-site-packages so the reused TileLang, torch and torch_npu stay
# exactly the ones the shared environment provides: this Adapter installs
# ForgeKernel on top, it does not install a second toolchain.
FK_ASCEND_VENV=

# ---------------------------------------------------------------------------
# Build selection — read by the engine
# ---------------------------------------------------------------------------
# The engine resolves a build directory name for every Adapter so it can take a
# resource lease. This Adapter builds nothing with Ninja, so the value is only
# the lease name; an empty value is a typo and is refused.
# FK_ASCEND_BUILD_SUBDIR=build

# ---------------------------------------------------------------------------
# Dependency policy — read by the Gate
# ---------------------------------------------------------------------------
# "frozen" (the default for recorded runs) installs only what is already
# present; "online" allows resolution. Changing this invalidates a recorded
# build, so it is a deliberate edit, not a flag.
FK_ASCEND_DEPENDENCY_POLICY=frozen

# Index used when the policy allows resolution. Leave the line out to use the
# documented public mirror; set it empty to install from no index at all.
# FK_ASCEND_PIP_INDEX=

# ---------------------------------------------------------------------------
# Agent configuration — accepted here, read by later gates
# ---------------------------------------------------------------------------
# The optimization loop is driven by an agent CLI. These name that CLI and its
# provider; the API KEY ITSELF is never stored here. The key is injected through
# the environment variable named by FK_ASCEND_PROVIDER_KEY_VARIABLE, so it does
# not appear in this file, in argv, or in a receipt.
FK_ASCEND_CLAUDE_BIN=
FK_ASCEND_PROVIDER_ENDPOINT=
FK_ASCEND_PROVIDER_MODEL=
FK_ASCEND_PROVIDER_KEY_VARIABLE=

# Policy forwarded to the agent. `off` keeps the sandbox decision the operator
# already made; it is not a security boundary and must not be described as one.
FK_ASCEND_AGENT_SANDBOX=off
FK_ASCEND_AGENT_NETWORK=online

# ---------------------------------------------------------------------------
# Illustrative shapes ONLY
# ---------------------------------------------------------------------------
# The commented lines below show what a value looks like. They are not values
# from any machine and must not be copied literally. Toolchain roots are written
# as <cann-root> / <user> rather than as this host's directories, because a
# committed template must not name any machine.
#
# FK_ASCEND_CANN_SETUP=<cann-root>/cann/set_env.sh
# FK_ASCEND_PYTHON=/home/<user>/.conda/envs/<env>/bin/python3
# FK_ASCEND_TILELANG_SOURCE=/home/<user>/workspace/tilelang
# FK_ASCEND_TILELANG_BUILD=/home/<user>/workspace/tilelang/build
# FK_ASCEND_WORK_ROOT=/home/<user>/workspace/<work-root>
# FK_ASCEND_VENV=/home/<user>/workspace/<work-root>/venv/fk
# FK_ASCEND_TILELANG_CACHE_DIR=/home/<user>/workspace/<work-root>/cache/tilelang
# FK_ASCEND_CLAUDE_BIN=/home/<user>/.local/bin/claude
# FK_ASCEND_PROVIDER_ENDPOINT=https://gateway.example.invalid
# FK_ASCEND_PROVIDER_MODEL=<provider-model-id>
# FK_ASCEND_PROVIDER_KEY_VARIABLE=MAGPIE_API_KEY
