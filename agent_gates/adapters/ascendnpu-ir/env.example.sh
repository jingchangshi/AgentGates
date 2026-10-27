# CANN and Conda for this machine — AscendNPU-IR Adapter.
#
# This is the single source of truth for the AscendNPU-IR toolchain. The Adapter
# reads it on every Gate, so editing it takes effect on the next Gate. No
# container recreation, and no reinstall of the Pack, is needed.
#
# Make it real by copying it next to itself. `env.sh` is git-ignored, so it is
# never committed and never leaves the machine:
#
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# That is the file the Adapter reads. It finds it by walking up from the running
# Gate to the checkout that owns it, so the same file serves a human and an
# agent working in the repository, whether the Gate runs from this checkout or
# from an installed Pack inside it. A Pack reinstall never overwrites it, because
# the manifest pins only `scripts/`.
#
# Two locations are honoured, in this order:
#
#     1. $ASCENDNPU_ENV_FILE, an explicit absolute path. It must exist.
#     2. adapters/ascendnpu-ir/env.sh, beside this template.
#
# There is no $HOME default: a machine that keeps the file elsewhere names it
# with ASCENDNPU_ENV_FILE, so no Gate reads a file nobody reviewed.
#
# Editing the file changes the next Gate. No container recreation and no Pack
# reinstall is needed.
#
# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
# Only plain `NAME=value` lines are read. `#` comments and blank lines are
# allowed. The Adapter parses the file as data and never executes it, so command
# substitution and variable expansion do not happen. Quotes around a value are
# optional and are stripped.
#
# The Adapter also reads this file when a human sources it, because the syntax is
# valid shell. Serving both is why the format stays this simple. Sourcing gives
# you the three values; to apply them to the current shell, source the committed
# initialization script next:
#
#     source <adapter>/env.sh
#     source <adapter>/scripts/init-toolchain.sh
#
# That script is the only place the CANN/Conda sequence is written, and it is the
# same file the Gate inlines into the container-side script it runs.
#
# All paths are CONTAINER paths, not host paths. They must be reachable through
# one of the bind mounts configured in docker/config.env. With the usual
# `/home:/home` mount the host and container path are identical.
#
# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
# CANN set_env.sh. Optional; leave empty on a machine without CANN. When set, the
# file must be readable inside the container and must define ASCEND_HOME_PATH, or
# the Gate fails closed.
#
# A relative path is an error. So is a path the container cannot read: the Gate
# reports it instead of running Ninja with a partially initialized environment.
ASCENDNPU_CANN_SETUP=

# Conda: conda.sh and the environment to activate. Optional. Set both or
# neither. Setting only one is an error, because the Gate must not silently run
# Ninja against a different Python than the build tree was configured with.
#
# Both stay empty for this image. It already provides cmake and ninja under
# /opt/build-tools, links them into /usr/local/bin, and puts /usr/local/bin on
# the container PATH; python3 is /usr/bin/python3. Set Conda only when a build
# tree was configured against a Conda interpreter.
ASCENDNPU_CONDA_SH=
ASCENDNPU_CONDA_ENV=

# ccec compiler bin directory. Read by scripts/full-build.sh, which needs it to
# configure a fresh build tree. A container path, reachable through a bind mount.
# The Gate ignores it: an incremental build never reconfigures, so it only needs
# an existing build.ninja. Leave empty until you run a full build.
ASCENDNPU_CCEC_BIN=
