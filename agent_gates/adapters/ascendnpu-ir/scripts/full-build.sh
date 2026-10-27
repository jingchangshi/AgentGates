#!/usr/bin/env bash
# Fresh full build for the AscendNPU-IR Adapter.
#
# This file is the only place the from-scratch build command is written. A human
# sources it inside the build container, after the toolchain is initialized:
#
#     . <checkout>/adapters/ascendnpu-ir/env.sh
#     . <checkout>/adapters/ascendnpu-ir/scripts/init-toolchain.sh
#     . <checkout>/adapters/ascendnpu-ir/scripts/full-build.sh
#     full_build
#
# The Gate is separate and does not call this file. `docker_gate.py` builds
# incrementally and requires an existing `build.ninja`; this script is what
# creates that build tree. The manifest pins it so the command cannot drift
# out of the repository unnoticed.
#
# Sourcing only defines the function. Unlike `init-toolchain.sh` there is no
# auto-run guard here: sourcing a file must never start a build. The call is
# yours to make.
#
# `full_build` reads three values from the calling shell at call time:
#
#     ASCENDNPU_CCEC_BIN       required. The ccec compiler bin directory, a
#                              container path reachable through a bind mount.
#     ASCENDNPU_BUILD_SUBDIR   build directory under the source root; default
#                              `build`.
#     ASCENDNPU_NINJA_JOBS     parallel jobs; default the CPU count.
#
# `ASCENDNPU_CCEC_BIN` is the one value the repository cannot know, because it
# names that machine's ccec toolchain, so a machine supplies it in its env.sh.
#
# A failure is returned, never exited, so sourcing cannot end an interactive
# session. `set -e`, `set -u` and `pipefail` are not changed in the calling
# shell: the pipeline runs in a subshell, so `$?` after the call is the build's
# status and the caller keeps its own options.

_full_build_error() {
    printf 'ascendnpu-ir-full-build: %s\n' "$*" >&2
}

full_build() {
    local source_root build_subdir ccec_bin jobs log status

    if [ $# -eq 0 ]; then
        source_root=$PWD
    elif [ $# -eq 1 ]; then
        source_root=$1
    else
        _full_build_error 'expected no arguments, or exactly one: the source root'
        return 1
    fi

    build_subdir=${ASCENDNPU_BUILD_SUBDIR:-build}
    ccec_bin=${ASCENDNPU_CCEC_BIN:-}
    jobs=${ASCENDNPU_NINJA_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
    log="$source_root/build.log"

    if [ -z "$ccec_bin" ]; then
        _full_build_error 'ASCENDNPU_CCEC_BIN is not set; add it to env.sh'
        return 1
    fi
    if [ ! -d "$source_root" ]; then
        _full_build_error "no such source root: $source_root"
        return 1
    fi
    if [ ! -r "$source_root/build-tools/build.sh" ]; then
        _full_build_error "no build-tools/build.sh under $source_root"
        return 1
    fi

    printf 'ascendnpu-ir-full-build: source=%s build=%s jobs=%s ccec=%s log=%s\n' \
           "$source_root" "$build_subdir" "$jobs" "$ccec_bin" "$log" >&2

    (
        set -o pipefail
        cd "$source_root" || exit 1
        bash ./build-tools/build.sh \
            --c-compiler clang \
            --cxx-compiler clang++ \
            --enable-lld \
            --build-type Release \
            --enable-assertion \
            -t \
            --bisheng-compiler "$ccec_bin" \
            --disable-werror \
            --disable-bishengir-werror \
            --build-triton \
            --build-shmem-template \
            --build "./$build_subdir" \
            -j "$jobs" \
            2>&1 | tee "$log"
    )
    status=$?

    if [ "$status" -ne 0 ]; then
        _full_build_error "build failed with status $status; see $log"
        return "$status"
    fi
    return 0
}
