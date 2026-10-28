# Machine configuration for this machine - AscendNPU-IR Adapter.
#
# This is the single source of truth for every machine fact this Adapter uses:
# the container it enters, the toolchain inside it, which build tree it builds,
# and how wide. The Adapter reads it on every Gate, so editing it takes effect on
# the next Gate. No container recreation, and no reinstall of the Pack, is needed.
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
# valid shell. Serving both is why the format stays this simple. Sourcing exports
# the container identity and the build selection; to initialize the toolchain in
# the current shell, source the committed initialization script next:
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

# ccec compiler bin directory. scripts/full-build.sh needs it to configure a
# fresh build tree. A container path, reachable through a bind mount.
#
# Read only when the build directory has neither build.ninja nor CMakeCache.txt:
# that is the from-zero branch. An incremental Gate never reconfigures, so it
# needs no compiler path. A machine that only ever builds incrementally may leave
# it empty; a full build refuses to start without it.
ASCENDNPU_CCEC_BIN=

# ---------------------------------------------------------------------------
# Container identity
# ---------------------------------------------------------------------------
# The container this Adapter enters. It is a machine fact, so it lives here
# beside the toolchain: the container docker/config.env created and the container
# a Gate enters cannot drift apart, and a stale export cannot point a Gate at a
# container nobody created.
#
# The name and user must be plain names; HOME must be an absolute container path.
# All three are required. They may also be exported, but only with the same
# value: a disagreeing export is refused rather than silently overridden, because
# neither order of precedence would show which one a Gate actually used.
ASCENDNPU_CONTAINER_NAME=
ASCENDNPU_CONTAINER_USER=
ASCENDNPU_CONTAINER_HOME=

# The container path of the source root, when the container sees the checkout at
# a different path than the bind-mount mapping derives. Leave the line out to use
# the derived mapping, which is what a /home:/home mount needs.
#
# It confirms a path; it cannot invent one. The mapping is still computed from
# `docker inspect` and compared, so a value that disagrees with the mounts is
# refused. It lives here rather than in the launching shell because it decides
# which tree the Gate runs in.
#
# ASCENDNPU_CONTAINER_SOURCE=/workspace/AscendNPU-IR

# ---------------------------------------------------------------------------
# Build selection
# ---------------------------------------------------------------------------
# The build tree this Adapter builds in, and how wide it builds. Both are machine
# facts for the same reason the container is: AgentGates takes its build lock and
# records its artifacts from the same value, so a lock on 'build' never covers a
# Gate writing 'build-fast'.
#
# BUILD_SUBDIR is one plain directory name under the source root; a path
# separator, '.' or '..' is refused, and so is an empty value -- unset it by
# leaving the line out, which gives 'build'.
#
# NINJA_JOBS is a positive integer. Left out, the container uses its own CPU
# count and cgroup limits.
#
# Both lines are commented out here so the template states the defaults instead
# of overriding them with an empty value:
#
# ASCENDNPU_BUILD_SUBDIR=build
# ASCENDNPU_NINJA_JOBS=64
