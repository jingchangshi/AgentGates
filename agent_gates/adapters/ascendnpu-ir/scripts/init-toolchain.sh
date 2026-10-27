#!/usr/bin/env bash
# Toolchain initialization for the AscendNPU-IR Adapter.
#
# This file is the only place the CANN and Conda sequence is written. Two
# callers share it:
#
#   * A human, on the Host or in the container:
#
#         . <checkout>/adapters/ascendnpu-ir/env.sh
#         . <checkout>/adapters/ascendnpu-ir/scripts/init-toolchain.sh
#
#     Sourcing runs the initialization in the current shell, so the CANN and
#     Conda state it builds is kept. Running this file with `bash` throws that
#     state away, so source it instead.
#
#   * The Gate, scripts/docker_gate.py. The Pack tree is not bind-mounted into
#     the build container, so the Gate inlines this file's text into the
#     container-side script it already sends, then calls `init_toolchain` with
#     the three values it validated on the Host.
#
# `init_toolchain` takes the three values in the order CANN set_env.sh, conda.sh,
# conda environment name. `-` or an empty string means "not set". With no
# arguments at all it reads the ASCENDNPU_* shell variables instead, which is
# what sourcing env.sh leaves in the shell. There is no third form: a caller
# either has the file's values or passes all three, so an argument list can
# never be mistaken for a complete configuration when it is short. It never
# exits the caller's shell: every failure is printed on stderr and returned as a
# non-zero status, so a sourced run cannot end an interactive session.
#
# The order is fixed. CANN must load first, because set_env.sh exports
# ASCEND_HOME_PATH, which the BiShengIR CMake files expand into the `ccec`
# template build flags. Conda is optional and both Conda values stay empty for
# this image, which already provides cmake, ninja and python3 under
# /opt/build-tools and links cmake and ninja into /usr/local/bin, already on the
# container PATH. Set Conda only when a build tree was configured against a
# Conda interpreter. The tool checks come last, so a tool that either step adds
# to PATH is found.
#
# `set -u` is deliberately not used: the vendor scripts read variables that are
# legitimately unset on first entry.

_init_error() {
    printf 'ascendnpu-ir-init: %s\n' "$*" >&2
}

init_toolchain() {
    local cann conda_sh conda_env

    if [ $# -eq 0 ]; then
        cann=${ASCENDNPU_CANN_SETUP:-}
        conda_sh=${ASCENDNPU_CONDA_SH:-}
        conda_env=${ASCENDNPU_CONDA_ENV:-}
    elif [ $# -eq 3 ]; then
        cann=$1
        conda_sh=$2
        conda_env=$3
    else
        _init_error 'expected no arguments, or exactly three: CANN set_env.sh, conda.sh, ' \
                    'conda env'
        return 1
    fi
    if [ "$cann" = '-' ]; then cann=; fi
    if [ "$conda_sh" = '-' ]; then conda_sh=; fi
    if [ "$conda_env" = '-' ]; then conda_env=; fi

    # Keep the PATH this shell already has; fall back only when it is unset or
    # empty. CANN and `conda activate` extend this PATH, so the tool checks must
    # look the tools up from it rather than from a fixed directory.
    export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

    if [ -n "$cann" ]; then
        if [ ! -r "$cann" ]; then
            _init_error "CANN setup not readable: $cann"
            return 1
        fi
        # shellcheck disable=SC1090
        if ! . "$cann"; then
            _init_error "sourcing CANN setup failed: $cann"
            return 1
        fi
        if [ -z "${ASCEND_HOME_PATH:-}" ]; then
            _init_error "CANN setup did not define ASCEND_HOME_PATH: $cann"
            return 1
        fi
    fi

    if { [ -n "$conda_sh" ] && [ -z "$conda_env" ]; } || { [ -z "$conda_sh" ] && [ -n "$conda_env" ]; }
    then
        _init_error 'ASCENDNPU_CONDA_SH and ASCENDNPU_CONDA_ENV must be provided together; ' \
                    'set both or neither'
        return 1
    fi
    if [ -n "$conda_sh" ]; then
        if [ ! -r "$conda_sh" ]; then
            _init_error "conda.sh not readable: $conda_sh"
            return 1
        fi
        # shellcheck disable=SC1090
        if ! . "$conda_sh"; then
            _init_error "sourcing conda.sh failed: $conda_sh"
            return 1
        fi
        if ! conda activate "$conda_env"; then
            _init_error "conda activate $conda_env failed"
            return 1
        fi
    fi

    if ! command -v cmake >/dev/null; then
        _init_error 'cmake not found in PATH after environment initialization'
        return 1
    fi
    if ! command -v ninja >/dev/null; then
        _init_error 'ninja not found in PATH after environment initialization'
        return 1
    fi
    return 0
}

# A human who executes or sources this file initializes their own shell now. When
# the Gate has inlined the text, BASH_SOURCE is empty, so only the function is
# defined and the Gate drives it with the values it validated on the Host.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    init_toolchain "$@"
fi
