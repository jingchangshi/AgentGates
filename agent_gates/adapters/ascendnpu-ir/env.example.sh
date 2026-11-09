# Machine configuration for this machine - AscendNPU-IR Adapter.
#
# This is the single source of truth for every machine fact this Adapter uses:
# the container it enters, the toolchain inside it, which build tree it builds,
# and how wide. The Adapter reads it on every Gate, so editing it takes effect on
# the next Gate. No container recreation, and no reinstall of the Pack, is needed.
#
# This Adapter has two routes and one machine file. `ascend-build` and
# `ascend-regression` enter a build container; `ascend-build-host` and
# `ascend-regression-host` run the same Generator on the Host that launched the
# Gate, with no Docker in the path. The file describes ONE machine, so a key is
# read only by the route that can use it:
#
#   * ASCENDNPU_CONTAINER_* is required only by the docker route. The host route
#     never reads it: a host-only machine leaves all three empty, and the host
#     configuration check does not require them.
#   * ASCENDNPU_CANN_SETUP, ASCENDNPU_CCEC_BIN, ASCENDNPU_BUILD_SUBDIR and
#     ASCENDNPU_NINJA_JOBS are read by both routes, and each route reads the
#     paths its own way. On the docker route they are CONTAINER paths; on the
#     host route they are HOST paths.
#
# One file can serve both routes only while its paths mean something in both,
# which is the usual case with a `/home:/home` mount. Otherwise keep two files
# and select one with ASCENDNPU_ENV_FILE - that is the supported arrangement, and
# each run reads exactly the file it was given.
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
# Paths name the route that reads them. On the docker route every path is a
# CONTAINER path, and it must be reachable through one of the bind mounts
# configured in docker/config.env. With the usual `/home:/home` mount the host
# and container path are identical. On the host route the same keys are HOST
# paths: no container and no mapping exists, so the value must name a location in
# the filesystem the Gate itself runs in.
#
# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
# CANN set_env.sh. Optional; leave empty on a machine without CANN. When set, the
# file must be readable by the route that sources it and must define
# ASCEND_HOME_PATH, or the Gate fails closed. On the docker route it must be
# readable inside the container; on the host route it must be readable by the
# operator who launched the Gate, and sourcing it is how the host route puts the
# CANN compilers the build needs on the effective PATH.
#
# A relative path is an error. So is a path the route cannot read: the Gate
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
# fresh build tree. A container path on the docker route, reachable through a
# bind mount; a host path on the host route.
#
# Read only when the build directory has neither build.ninja nor CMakeCache.txt:
# that is the from-zero branch. An incremental Gate never reconfigures, so it
# needs no compiler path. A machine that only ever builds incrementally may leave
# it empty; a full build refuses to start without it.
#
# The host route is stricter here than the container ever had to be: it checks
# that the directory holds both `ccec` and `llvm-link` and that both are
# executable, because CMake only tests that they exist and a mode-only mistake
# would otherwise surface minutes later inside the toolchain.
ASCENDNPU_CCEC_BIN=

# ---------------------------------------------------------------------------
# Container identity
# ---------------------------------------------------------------------------
# The container the docker route enters. It is a machine fact, so it lives here
# beside the toolchain: the container docker/config.env created and the container
# a Gate enters cannot drift apart, and a stale export cannot point a Gate at a
# container nobody created.
#
# Required ONLY by the docker route. The host route never reads these three keys,
# so a host-only machine leaves them empty and its configuration check passes
# without them. They are not a host fact, and inventing a value for them would
# suggest a container exists.
#
# They are still allowed to sit in this file on a two-route machine: one file
# serves both routes when it describes both, and the host recipe simply ignores
# what it does not read.
#
# The name and user must be plain names; HOME must be an absolute container path.
# On the docker route all three are required. They may also be exported, but only
# with the same value: a disagreeing export is refused rather than silently
# overridden, because neither order of precedence would show which one a Gate
# actually used.
ASCENDNPU_CONTAINER_NAME=
ASCENDNPU_CONTAINER_USER=
ASCENDNPU_CONTAINER_HOME=

# The container path of the source root, when the container sees the checkout at
# a different path than the bind-mount mapping derives. Leave the line out to use
# the derived mapping, which is what a /home:/home mount needs. Docker route only:
# the host route runs in the source root it was given, so it has no path to
# confirm.
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
# Gate writing 'build-fast'. Both routes read these two keys, and both resolve
# BUILD_SUBDIR under the source root of the route - the container's source root,
# or the Host checkout for a host Gate.
#
# Because the build tree is one name under one source root, the two routes can
# share it when they share the root. Pointing them at different roots means
# pointing them at different trees: name the location in ASCENDNPU_ENV_FILE
# rather than expecting one file to mean two machines.
#
# BUILD_SUBDIR is one plain directory name under the source root; a path
# separator, '.' or '..' is refused, and so is an empty value -- unset it by
# leaving the line out, which gives 'build'.
#
# NINJA_JOBS is a positive integer. Left out, the route uses its own CPU count:
# the container's cgroup limits on the docker route, `nproc` on the host route.
#
# Both lines are commented out here so the template states the defaults instead
# of overriding them with an empty value:
#
# ASCENDNPU_BUILD_SUBDIR=build
# ASCENDNPU_NINJA_JOBS=64

# ---------------------------------------------------------------------------
# Host route: what the file cannot say
# ---------------------------------------------------------------------------
# The host route reads the same keys, but three of its requirements are about the
# machine rather than the file, so no key here can express them:
#
# Tool versions. Docker mode inherits its toolchain from the image. On the Host
# the toolchain is the operator's, and a stock install can be too old without
# saying so: the source requires ninja >= 1.12 (CMakeLists.txt:6-13) and
# cmake >= 3.28 (CMakeLists.txt:2). The host configuration check resolves each
# tool from the EFFECTIVE PATH - after CANN and Conda have had their say - and
# prints the path it found beside the version, because "the first ninja on PATH
# is /usr/bin/ninja" is what makes an old version actionable.
#
# The check also requires bash, timeout, nproc, clang, clang++, g++, python3,
# ld.lld and git, and fails if the current directory is not inside the git
# work tree that build-tools/build.sh expects.
#
# Submodules. Neither key in this file can supply them: a from-zero host build
# additionally needs third-party/llvm-project and third-party/shmem populated
# under the source root, which the build check reports at t=0 instead of after a
# long configure. An existing build tree needs neither.
#
# A missing cmake or ninja by NAME is reported first by the inlined
# scripts/init-toolchain.sh, with its own message; the version floors above are
# the part only this Adapter's check can state.
