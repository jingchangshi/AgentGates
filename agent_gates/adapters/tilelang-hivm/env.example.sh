# CANN and Conda for this machine — TileLang HIVM Adapters.
#
# This is the single configuration file for the TileLang toolchain. Three
# readers use it and nothing else does:
#
#   * scripts/docker_gate.py    — the build and regression Gates
#   * scripts/npu_gate.py       — the device Gate of the -npu Adapter
#   * scripts/setup-python-env.sh — the one-time Python provisioning
#   * scripts/full-build.sh     — `full_build`, a shell function a human calls
#
# Both Adapters read this one file because both use the `TILELANG_HIVM_`
# prefix; `tilelang-hivm-npu` has no template of its own. Editing it takes
# effect on the next Gate: no container recreation and no Pack reinstall.
#
# ---------------------------------------------------------------------------
# Create the file
# ---------------------------------------------------------------------------
# An absent file is not an error: it means "configure nothing". Both CANN and
# Conda are optional, so a Gate then runs with neither, which is correct on a
# machine without them and silently wrong on a machine with them. A build then
# fails deep inside pytest with `ImportError: libhccl.so` instead of at init.
# So on a machine that has CANN or Conda, create this file:
#
#     cd <checkout>/adapters/tilelang-hivm
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
#     1. $TILELANG_HIVM_ENV_FILE, an explicit absolute path. It must exist.
#     2. adapters/tilelang-hivm/env.sh, beside this template.
#
# The second is found by walking up from the Gate to the checkout that owns it,
# so the same file serves a human and an agent working in this repository,
# whether the Gate runs from this checkout or from an installed Pack inside it.
#
# ---------------------------------------------------------------------------
# This file is the only source of these values
# ---------------------------------------------------------------------------
# Exporting any TILELANG_HIVM_* here-documented key in the launching shell has
# NO effect: the Gates read them from this file only, so a stale export can
# never mask an edit. The single exception is documented in the provisioning
# script, and applies to that script alone.
#
# Container identity (`TILELANG_HIVM_CONTAINER_NAME`, `_CONTAINER_USER`,
# `_CONTAINER_HOME`) IS in this file, because it describes one machine's
# container exactly as its CANN and Conda paths describe its toolchain. It must
# name the container `docker/config.env` created; putting the two in one place
# is what stops a Gate from addressing a stopped container while the machine
# file names the live one.
#
# The Late-hivm compiler (`TILELANG_HIVM_HIVM_COMPILER`, `_HIVM_TARGET`) is also
# here. TileLang reads it as `HIVM_COMPILER` from the process environment, and
# `docker exec` inherits nothing, so the Gate forwards it explicitly. An export
# of `HIVM_COMPILER` in the launching shell is therefore ignored as well: the
# compiler a Gate measures is the one this file names.
#
# `TILELANG_HIVM_DEVICE_ID` is NOT in this file. It authorizes one Gate run, not
# a machine, so it stays an export of the shell that launches the device Gate.
#
# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
# Only plain `NAME=value` lines are read. `#` comments and blank lines are
# allowed. The Gate parses the file as data and never executes it, so command
# substitution and variable expansion do not happen. Quotes around a value are
# optional and are stripped.
#
# The syntax is valid shell, so a human can source this file to get the same
# paths, then source the committed initialization script to apply them:
#
#     source <checkout>/adapters/tilelang-hivm/env.sh
#     source <checkout>/adapters/tilelang-hivm/scripts/init-toolchain.sh
#
# That script is the only place the CANN/Conda sequence is written, and it is
# the same file the Gate inlines into the container-side script it runs.
#
# All paths are CONTAINER paths, not host paths. They must be reachable through
# one of the container's bind mounts; with the usual `/home:/home` and
# `/data:/data` mounts the host and container path are identical.

# ---------------------------------------------------------------------------
# Toolchain — read by both Gates
# ---------------------------------------------------------------------------
# CANN set_env.sh. Optional; leave empty on a machine without CANN. When set,
# the file must be readable inside the container and must define
# ASCEND_HOME_PATH, or the Gate fails closed. CANN loads first, because
# `import tilelang` loads `torch_npu`, whose native libraries resolve libhccl.so
# from the CANN layout.
TILELANG_HIVM_CANN_SETUP=

# Conda: conda.sh and the environment that provides pytest, torch and
# torch_npu. Optional. Set both or neither; setting only one is an error,
# because the Gate must not silently run pytest against a different
# interpreter.
#
# This names an installation that already exists. The provisioning script
# reuses it and never installs a second Conda while these are set.
TILELANG_HIVM_CONDA_SH=
TILELANG_HIVM_CONDA_ENV=

# Where Conda searches for environments. Required whenever the environment does
# not live under the installation prefix, which is the normal case for a user's
# own environments in `~/.conda/envs`. Without it `conda activate` fails with
# `EnvironmentNameNotFound: Could not find conda environment: <name>` even
# though `conda env list` shows the environment. One absolute directory.
TILELANG_HIVM_CONDA_ENVS_PATH=

# ---------------------------------------------------------------------------
# Late-hivm compiler — forwarded into the container
# ---------------------------------------------------------------------------
# TileLang's HIVM codegen (`tilelang/contrib/hivm.py`) reads `HIVM_COMPILER`
# from the process environment. `docker exec` starts a process that inherits
# nothing from the Gate, so the Gate forwards this value explicitly under the
# bare name `HIVM_COMPILER`. Without it `testing/ascend/codegen/test_hivm_build.py`
# skips every test that needs the compiler: the run stays green while proving
# nothing about codegen. Set it on a machine that has the compiler, and leave it
# empty on one that does not. The path is a CONTAINER path and must be an
# executable there; the Gate refuses to run when it is set but not executable.
TILELANG_HIVM_HIVM_COMPILER=

# The device the compile targets, forwarded as `HIVM_TARGET`. Optional. TileLang
# treats it as an assertion rather than an override: the emitted module, the
# compiler's `--target` and the output ELF must all name ONE device, so the Gate
# accepts only the codegen's own default (`Ascend950PR_9579`) and refuses any
# other value on the Host. Leave it empty to accept that default.
TILELANG_HIVM_HIVM_TARGET=

# ---------------------------------------------------------------------------
# Container identity — read by both Gates
# ---------------------------------------------------------------------------
# The container the Gates enter. It must be the container you created from
# `docker/config.env` in this directory, and those three values must match that
# file's unprefixed `CONTAINER_NAME`, `CONTAINER_USER` and `CONTAINER_HOME`.
# The Gates cannot read `config.env` (it describes creation, not use), so this
# file is where the identity is stated once for the whole toolchain.
#
# A Gate refuses to run when any of these is empty: it will not guess a
# container, and it will not fall back to a shell export.
#
# `TILELANG_HIVM_CONTAINER_USER` is the user the build Gate runs as. The device
# Gate ignores it and runs as root, because the driver refuses a non-root opener
# once any namespace holds the card.
TILELANG_HIVM_CONTAINER_NAME=
TILELANG_HIVM_CONTAINER_USER=
# HOME inside the container. Normally covered by the /home bind mount.
TILELANG_HIVM_CONTAINER_HOME=

# The container path of this checkout, when the container sees it at a different
# path than the bind-mount mapping derives. Leave the line out to use the derived
# mapping, which is what a /home:/home mount needs.
#
# It confirms a path; it cannot invent one. The mapping is still computed from
# `docker inspect` and compared, so a value that disagrees with the mounts is
# refused. It lives here rather than in the launching shell because it decides
# which tree both Gates run in.
#
# TILELANG_HIVM_CONTAINER_SOURCE=/workspace/tilelang

# ---------------------------------------------------------------------------
# Build selection
# ---------------------------------------------------------------------------
# Which tree this Adapter builds in, and how wide. Both are machine facts for the
# same reason the container is: AgentGates takes its build lock and records its
# artifacts from this same file, so a lock on 'build' never covers a Gate writing
# 'build-fast'.
#
# BUILD_SUBDIR is one plain directory name under the source root; a path
# separator, '.' or '..' is refused, and so is an empty value -- unset it by
# leaving the line out, which gives 'build'.
#
# NINJA_JOBS is a positive integer. Left out, the container uses its own CPU
# count and cgroup limits.
#
# TILELANG_HIVM_BUILD_SUBDIR=build
# TILELANG_HIVM_NINJA_JOBS=64

# ---------------------------------------------------------------------------
# Provisioning — read only by scripts/setup-python-env.sh
# ---------------------------------------------------------------------------
# Both Gates accept these keys and ignore them, so one file serves all three
# readers. They are here rather than in the calling shell because a shell export
# leaves no record: it is invisible in review, lost when the shell exits, and a
# stale export silently provisions something other than what this file
# documents.

# Vendor wheel directory, holding `torch-<v>+cpu` and `torch_npu-<v>`. Required
# by the provisioning script only. A CUDA `torch` build is refused, because
# those wheels cannot run on this host.
TILELANG_HIVM_WHEEL_DIR=

# Index for the small pure-Python dependencies. Leave the line out to use the
# documented public mirror; set it to an empty value to install from no index at
# all, which is what an offline run needs.
# TILELANG_HIVM_PIP_INDEX=

# Interpreter version for a newly created environment. Ignored when the
# environment already exists.
TILELANG_HIVM_PYTHON_VERSION=3.11

# TileLang checkout for the editable install. Empty means the current directory.
TILELANG_HIVM_SOURCE_ROOT=

# ---------------------------------------------------------------------------
# Illustrative shapes ONLY
# ---------------------------------------------------------------------------
# The commented lines below show what a value looks like. They are not values
# from any machine and must not be copied literally.
#
# TILELANG_HIVM_CONTAINER_NAME=tilelang-hivm-dev
# TILELANG_HIVM_CONTAINER_USER=builder
# TILELANG_HIVM_CONTAINER_HOME=/home/builder
# TILELANG_HIVM_CANN_SETUP=/opt/cann/cann-9.x/set_env.sh
# TILELANG_HIVM_CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
# TILELANG_HIVM_CONDA_ENV=py311
# TILELANG_HIVM_CONDA_ENVS_PATH=/home/u/.conda/envs
# TILELANG_HIVM_WHEEL_DIR=/opt/wheels
# TILELANG_HIVM_HIVM_COMPILER=/opt/ascendnpu-ir/bin/bishengir-compile
# TILELANG_HIVM_HIVM_TARGET=Ascend950PR_9579
