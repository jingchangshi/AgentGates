#!/usr/bin/env bash
# Toolchain initialization for the TileLang HIVM Adapters.
#
# This file is the only place the CANN and Conda sequence is written. Two
# callers share it:
#
#   * A human, on the Host or in the container:
#
#         . <checkout>/adapters/tilelang-hivm/env.sh
#         . <checkout>/adapters/tilelang-hivm/scripts/init-toolchain.sh
#
#     Sourcing runs the initialization in the current shell, so the CANN and
#     Conda state it builds is kept. Running this file with `bash` throws that
#     state away, so source it instead.
#
#   * The Gate, scripts/docker_gate.py. The Pack tree is not bind-mounted into
#     the build container, so the Gate inlines this file's text into the
#     container-side script it already sends, then calls `init_toolchain` with
#     the four values it validated on the Host.
#
# `init_toolchain` takes four values in the order CANN set_env.sh, conda.sh,
# conda environment name, conda environments directory. `-` or an empty string
# means "not set". With no arguments at all it reads the TILELANG_HIVM_* shell
# variables instead, which is what sourcing env.sh leaves in the shell. There is
# no third form: a caller either has the file's values or passes all four, so an
# argument list can never be mistaken for a complete configuration when it is
# missing the environments directory. It never exits the caller's shell: every
# failure is printed on stderr and returned as a non-zero status, so a sourced
# run cannot end an interactive session.
#
# The order is fixed. CANN must load first, because `import tilelang` loads
# `torch_npu`, whose native libraries resolve `libhccl.so` from the CANN layout
# that `set_env.sh` puts on the loader path. Loading Conda first would let its
# activation fix the loader and PATH state before CANN prepends its library
# directories, and the damage would surface only much later as an ImportError
# inside pytest. Conda is optional; it supplies the interpreter that provides
# pytest, torch and torch_npu. The tool checks come last, so an interpreter that
# either step adds to PATH is found.
#
# `set -u` is deliberately not used: the vendor scripts read variables that are
# legitimately unset on first entry.

_init_error() {
    printf 'tilelang-hivm-init: %s\n' "$*" >&2
}

init_toolchain() {
    local cann conda_sh conda_env conda_envs_path

    if [ $# -eq 0 ]; then
        cann=${TILELANG_HIVM_CANN_SETUP:-}
        conda_sh=${TILELANG_HIVM_CONDA_SH:-}
        conda_env=${TILELANG_HIVM_CONDA_ENV:-}
        conda_envs_path=${TILELANG_HIVM_CONDA_ENVS_PATH:-}
    elif [ $# -eq 4 ]; then
        cann=$1
        conda_sh=$2
        conda_env=$3
        conda_envs_path=$4
    else
        _init_error 'expected no arguments, or four: CANN set_env.sh, conda.sh, ' \
                    'conda env, conda envs path'
        return 1
    fi
    if [ "$cann" = '-' ]; then cann=; fi
    if [ "$conda_sh" = '-' ]; then conda_sh=; fi
    if [ "$conda_env" = '-' ]; then conda_env=; fi
    if [ "$conda_envs_path" = '-' ]; then conda_envs_path=; fi

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
        _init_error 'TILELANG_HIVM_CONDA_SH and TILELANG_HIVM_CONDA_ENV must be provided together; ' \
                    'set both or neither'
        return 1
    fi
    if [ -n "$conda_envs_path" ] && [ -z "$conda_sh" ]; then
        _init_error 'TILELANG_HIVM_CONDA_ENVS_PATH needs TILELANG_HIVM_CONDA_SH and ' \
                    'TILELANG_HIVM_CONDA_ENV'
        return 1
    fi
    # A Conda installation searches only the environments under its own prefix.
    # A user's environments normally live in `~/.conda/envs`, so without this
    # value `conda activate` fails with `EnvironmentNameNotFound: Could not find
    # conda environment: <name>`. Set it before `conda.sh` is sourced, because
    # that script is what reads it.
    if [ -n "$conda_envs_path" ]; then
        export CONDA_ENVS_PATH="$conda_envs_path${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
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

    # The TileLang tool check is the interpreter, not a build tool. cmake and
    # ninja are not checked: the build tree is already configured, so the Gate
    # never reconfigures, and a from-scratch build (scripts/full-build.sh)
    # checks them itself.
    if ! command -v python3 >/dev/null; then
        _init_error 'python3 not found in PATH after environment initialization'
        return 1
    fi
    # pytest is reported, not enforced. The regression Gate needs it (`python3 -m
    # pytest`), but the build Gate does not, so refusing here would break a build
    # that is valid. The import probe only gives the failure a name before the
    # Gate runs; a missing pytest still fails the regression branch itself.
    if ! python3 -c 'import pytest' >/dev/null 2>&1; then
        _init_error 'note: python3 cannot import pytest; only the regression Gate needs it'
    fi
    return 0
}

# A human who executes or sources this file initializes their own shell now. When
# the Gate has inlined the text, BASH_SOURCE is empty, so only the function is
# defined and the Gate drives it with the values it validated on the Host.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    init_toolchain "$@"
fi
