#!/usr/bin/env bash
# Fresh full build for the TileLang HIVM Adapter.
#
# This file is the only place the from-scratch build command is written. A human
# sources it inside the build container, after the toolchain is initialized:
#
#     . <checkout>/adapters/tilelang-hivm/env.sh
#     . <checkout>/adapters/tilelang-hivm/scripts/init-toolchain.sh
#     . <checkout>/adapters/tilelang-hivm/scripts/full-build.sh
#     full_build
#
# The Gate is separate and does not call this file. `docker_gate.py` builds
# incrementally and requires an existing `build.ninja`; this script is what
# creates that build tree. The manifest pins it so the command cannot drift out
# of the repository unnoticed.
#
# Sourcing only defines the function. Unlike `init-toolchain.sh` there is no
# auto-run guard here: sourcing a file must never start a build. The call is
# yours to make.
#
# `full_build` reads three optional values from the calling shell at call time:
#
#     TILELANG_HIVM_BUILD_SUBDIR  build directory name under the source root;
#                                 default `build`. A single name: no `/`, no `..`.
#     TILELANG_HIVM_NINJA_JOBS    parallel jobs; default bounded by memory, see
#                                 below.
#     TILELANG_HIVM_NINJA_JOB_MEMORY_MB
#                                 peak memory one compile job is assumed to need;
#                                 default 2048.
#     TILELANG_HIVM_CMAKE_ARGS    extra arguments appended to the configure step;
#                                 default empty.
#
# No CANN or Conda variable is required here. CANN and Conda are applied by
# `init-toolchain.sh`, which must have been sourced first: the compiler
# toolchain (`cmake`, `ninja`, `gcc`) comes from the image, bisheng from CANN and
# the interpreter from Conda. The precondition is checked, not assumed.
#
# An existing build directory is reconfigured in place and never deleted:
# `cmake -S . -B <build>` is idempotent. For a genuinely fresh tree, remove the
# directory yourself first; this script does not delete anything.
#
# A failure is returned, never exited, so sourcing cannot end an interactive
# session. `set -e`, `set -u` and `pipefail` are not changed in the calling
# shell: the pipeline runs in a subshell, so `$?` after the call is the build's
# status and the caller keeps its own options.

_full_build_error() {
    printf 'tilelang-hivm-full-build: %s\n' "$*" >&2
}

# Print how many compile jobs this machine can hold at once.
#
# The CPU count is not a usable default. One TileLang translation unit peaks at
# several hundred megabytes, and a 256-core server running 256 of them is
# routinely killed by the OOM killer with `c++: fatal error: Killed signal
# terminated program cc1plus` halfway through the build, which reads as a source
# error and costs the whole build. The bound is therefore the memory available to
# this container divided by the assumed per-job peak: fewer jobs on a small
# machine, and on a large one still fewer than the core count. When neither the
# memory nor an explicit value is readable, assume a conservative four.
_full_build_default_jobs() {
    local job_memory_mb=$1 cores limit_mb by_cpu by_memory

    cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
    case "$cores" in ''|*[!0-9]*) cores=4 ;; esac

    # MemAvailable, not MemTotal: this is a shared server, and the build may only
    # use what nobody else is using right now. MemTotal would size the build for a
    # machine that is already busy and OOM it against the other tenants.
    limit_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null)
    case "$limit_mb" in ''|*[!0-9]*|0) printf '%s\n' "$cores"; return 0 ;; esac

    case "$job_memory_mb" in ''|*[!0-9]*|0) printf '%s\n' "$cores"; return 0 ;; esac

    by_cpu=$cores
    # Leave a quarter of the memory for the linker, the Python parent and the
    # page cache; a build that fits exactly is a build that swaps.
    by_memory=$(( limit_mb * 3 / 4 / job_memory_mb ))
    [ "$by_memory" -ge 1 ] || by_memory=1

    if [ "$by_memory" -lt "$by_cpu" ]; then
        printf '%s\n' "$by_memory"
    else
        printf '%s\n' "$by_cpu"
    fi
}

full_build() {
    local source_root build_subdir jobs cmake_args log status

    if [ $# -eq 0 ]; then
        source_root=$PWD
    elif [ $# -eq 1 ]; then
        source_root=$1
    else
        _full_build_error 'expected no arguments, or exactly one: the source root'
        return 1
    fi

    build_subdir=${TILELANG_HIVM_BUILD_SUBDIR:-build}
    local job_memory_mb=${TILELANG_HIVM_NINJA_JOB_MEMORY_MB:-2048}
    jobs=${TILELANG_HIVM_NINJA_JOBS:-$(_full_build_default_jobs "$job_memory_mb")}
    cmake_args=${TILELANG_HIVM_CMAKE_ARGS:-}
    log="$source_root/build.log"

    case "$build_subdir" in
        ''|.|..|*/*)
            _full_build_error "TILELANG_HIVM_BUILD_SUBDIR must be one plain directory name: $build_subdir"
            return 1
            ;;
    esac
    if [ ! -d "$source_root" ]; then
        _full_build_error "no such source root: $source_root"
        return 1
    fi
    if [ ! -r "$source_root/CMakeLists.txt" ]; then
        _full_build_error "no CMakeLists.txt under $source_root"
        return 1
    fi
    if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
        _full_build_error 'cmake and ninja must be on PATH; source scripts/init-toolchain.sh first'
        return 1
    fi

    printf 'tilelang-hivm-full-build: source=%s build=%s jobs=%s cmake_args=%s log=%s\n' \
           "$source_root" "$build_subdir" "$jobs" "${cmake_args:-<none>}" "$log" >&2

    (
        set -o pipefail
        cd "$source_root" || exit 1
        # shellcheck disable=SC2086  # TILELANG_HIVM_CMAKE_ARGS is a deliberate word list
        cmake -S . -B "./$build_subdir" -G Ninja \
              -DCMAKE_BUILD_TYPE=Release -DUSE_ASCEND=ON -DUSE_CUDA=OFF \
              $cmake_args \
              2>&1 | tee "$log" || exit $?
        ninja -C "./$build_subdir" -j "$jobs" tilelang \
              2>&1 | tee -a "$log"
    )
    status=$?

    if [ "$status" -ne 0 ]; then
        _full_build_error "build failed with status $status; see $log"
        return "$status"
    fi
    return 0
}
