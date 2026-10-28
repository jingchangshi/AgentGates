#!/usr/bin/env bash
# Full build for the AscendNPU-IR Adapter, from an empty build directory.
#
# This file is the only place the from-scratch build command is written. Two
# callers share it:
#
#   * A human, inside the build container, after the toolchain is initialized:
#
#         . <checkout>/adapters/ascendnpu-ir/env.sh
#         . <checkout>/adapters/ascendnpu-ir/scripts/init-toolchain.sh
#         . <checkout>/adapters/ascendnpu-ir/scripts/full-build.sh
#         full_build                       # or: full_build /abs/source
#
#   * The Gate, scripts/docker_gate.py, when the configured build directory has
#     neither `build.ninja` nor `CMakeCache.txt`. The Pack tree is not
#     bind-mounted into the build container, so the Gate inlines this file's text
#     and calls `full_build` with the values it validated on the Host. That is
#     why the from-zero path a Gate exercises and the command a human runs are
#     one reviewed, digest-pinned file rather than two copies that can drift.
#
# Sourcing only defines the function: sourcing a file must never start a build.
#
# `full_build [SOURCE_ROOT] [EXTRA...]` reads these values:
#
#     ASCENDNPU_CCEC_BIN        required. ccec compiler bin directory, a container
#                               path reachable through a bind mount. It is the one
#                               value the repository cannot know, so the machine
#                               supplies it in env.sh.
#     ASCENDNPU_BUILD_SUBDIR    build directory under the source root; default
#                               `build`.
#     ASCENDNPU_NINJA_JOBS      parallel jobs; default the CPU count. The job
#                               count is what the user's own command passes as
#                               `-j 64`; it is a machine fact, so env.sh holds
#                               it.
#     ASCENDNPU_FULL_BUILD_LOG  where to tee the log; default
#                               `<source>/build.log`. An existing file is never
#                               overwritten: the log then becomes
#                               `build-<UTC>.log` beside it, so a previous
#                               measurement survives. `-` means stdout only,
#                               which is what the Gate uses: a Gate writes no file
#                               into the source tree.
#
# EXTRA arguments are forwarded to `build-tools/build.sh` unchanged, so a human
# can add a flag the Adapter does not model. `-r`/`--rebuild` is refused: it
# deletes the configured build directory, and no Adapter - manual helper or Gate -
# may remove a tree it did not create. `--build`, `-j`, `-t` and
# `--bisheng-compiler` are supplied below and may not be repeated.
#
# The default command is the one this machine's operators use:
#
#     bash ./build-tools/build.sh \
#         --c-compiler clang --cxx-compiler clang++ --enable-lld \
#         --build-type Release --enable-assertion -t \
#         --bisheng-compiler "$ASCENDNPU_CCEC_BIN" \
#         --disable-werror --disable-bishengir-werror \
#         --build-triton --build-shmem-template --build "./$build_subdir" \
#         -j "$jobs"
#
# A failure is returned, never exited, so sourcing cannot end an interactive
# session. `set -e`, `set -u` and `pipefail` are not changed in the calling
# shell: the pipeline runs in a subshell, so `$?` after the call is the build's
# status and the caller keeps its own options.

_full_build_error() {
    printf 'ascendnpu-ir-full-build: %s\n' "$*" >&2
}

full_build() {
    local source_root build_subdir ccec_bin jobs log status arg extra
    local -a extras

    source_root=
    extras=()
    for arg in "$@"; do
        if [ -z "$source_root" ]; then
            source_root=$arg
        else
            extras+=("$arg")
        fi
    done

    if [ -z "$source_root" ]; then
        source_root=$PWD
    fi

    build_subdir=${ASCENDNPU_BUILD_SUBDIR:-build}
    ccec_bin=${ASCENDNPU_CCEC_BIN:-}
    jobs=${ASCENDNPU_NINJA_JOBS:-}
    log=${ASCENDNPU_FULL_BUILD_LOG:-$source_root/build.log}

    if [ -z "$ccec_bin" ]; then
        _full_build_error 'ASCENDNPU_CCEC_BIN is not set; add it to env.sh'
        return 1
    fi
    if [ -z "$jobs" ]; then
        jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
    fi
    if [ ! -d "$source_root" ]; then
        _full_build_error "no such source root: $source_root"
        return 1
    fi
    if [ ! -r "$source_root/build-tools/build.sh" ]; then
        _full_build_error "no build-tools/build.sh under $source_root"
        return 1
    fi

    for arg in "${extras[@]}"; do
        case "$arg" in
            -r|--rebuild)
                _full_build_error "$arg would delete $source_root/$build_subdir; cleaning a build directory is not this helper's decision"
                return 1
                ;;
            --c-compiler|--cxx-compiler|--enable-lld|--build-type|--enable-assertion|\
            -t|--build-bishengir-template|--bisheng-compiler|--disable-werror|\
            --disable-bishengir-werror|--build-triton|--disable-build-triton|\
            --build-shmem-template|-o|--build|-j|--jobs)
                _full_build_error "$arg is supplied by this helper and may not be repeated"
                return 1
                ;;
        esac
    done

    # Never overwrite a log a previous run left behind: the earlier measurement
    # is evidence, and silently replacing it loses the only record of that run.
    if [ "$log" != '-' ] && [ -n "$log" ]; then
        if [ -e "$log" ]; then
            log="${log%.log}-$(date -u +%Y%m%dT%H%M%SZ).log"
        fi
    fi

    printf 'ascendnpu-ir-full-build: source=%s build=%s jobs=%s ccec=%s log=%s\n' \
           "$source_root" "$build_subdir" "$jobs" "$ccec_bin" "$log" >&2

    (
        set -o pipefail
        cd "$source_root" || exit 1
        if [ "$log" = '-' ] || [ -z "$log" ]; then
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
                "${extras[@]}"
        else
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
                "${extras[@]}" \
                2>&1 | tee "$log"
        fi
    )
    status=$?

    if [ "$status" -ne 0 ]; then
        _full_build_error "build failed with status $status; see $log"
        return "$status"
    fi
    return 0
}
