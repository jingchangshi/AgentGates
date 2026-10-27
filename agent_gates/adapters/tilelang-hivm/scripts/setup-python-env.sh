#!/usr/bin/env bash
# Create the Python environment the TileLang HIVM Gates run in.
#
# This file is the only place the Python dependency set is written. Run it once
# inside the build container as the container user. Root is never needed: the
# Conda installation it uses already exists and is writable by that user, and
# nothing here installs system packages.
#
#     bash <checkout>/adapters/tilelang-hivm/scripts/setup-python-env.sh
#
# The configuration is the machine file `adapters/tilelang-hivm/env.sh`, the same
# file both TileLang Gates read. Start from the committed template:
#
#     cp adapters/tilelang-hivm/env.example.sh adapters/tilelang-hivm/env.sh
#     chmod 600 adapters/tilelang-hivm/env.sh
#
# Then set the values this script needs:
#
#     TILELANG_HIVM_WHEEL_DIR          directory holding the vendor wheels; required
#     TILELANG_HIVM_CONDA_SH           conda.sh of the installation to use; required
#     TILELANG_HIVM_CONDA_ENV          environment to create or reuse
#     TILELANG_HIVM_CONDA_ENVS_PATH    extra directory Conda searches for
#                                      environments, normally ~/.conda/envs
#     TILELANG_HIVM_PYTHON_VERSION     interpreter version; default 3.11
#     TILELANG_HIVM_PIP_INDEX          index for the small dependencies; default a
#                                      public PyPI mirror, empty string disables it
#     TILELANG_HIVM_SOURCE_ROOT        TileLang checkout for the editable
#                                      install; default the current directory
#     TILELANG_HIVM_CANN_SETUP         CANN set_env.sh, needed by the final check
#
# The file is the primary mechanism because a shell export leaves no record: it
# is invisible in review, lost when the shell exits, and a stale export silently
# provisions something other than what the machine is documented to have. Two
# exceptions, both narrow:
#
#   * `TILELANG_HIVM_ENV_FILE` selects which file to read. A bootstrap value
#     cannot come from the file it names.
#   * Any other `TILELANG_HIVM_*` value exported in the calling shell overrides
#     the file for this one run, and the script prints which values it overrode.
#     This exists for a one-off run and for the Adapter's own tests; it is not
#     how a machine is configured.
#
# The Conda installation is reused, never created. `TILELANG_HIVM_CONDA_SH` names
# the installation the Gates source, so the environment this script builds is the
# environment the Gates run in. Installing a second Conda would create one the
# machine file does not mention.
#
# The exit status is 0 only when the environment imports `tilelang` and reports an
# available NPU. Anything else is printed on stderr and returns non-zero.
#
# Why the wheels are local files rather than an index download: this is an Ascend
# server. `pip install torch` from PyPI resolves to the CUDA build and pulls
# several gigabytes of `nvidia-*` wheels that cannot run here. The vendor CI
# directory ships `torch-<v>+cpu` and `torch_npu-<v>`, which are the pair the
# Gate has been verified against. Small pure-Python dependencies still come from
# an index, which is why this script needs network access once.

set -o pipefail

_setup_error() {
    printf 'tilelang-hivm-setup-python: %s\n' "$*" >&2
}

_setup_die() {
    _setup_error "$@"
    exit 1
}

# The strict `NAME=value` grammar the Gates use. A machine file is data, so
# reading it never executes Host code. Kept in step with `_parse_env_file` in
# `scripts/docker_gate.py`; a value that the Gate rejects must not be accepted
# here, or the two disagree about what the machine is configured to do.
_env_file_lookup() {
    local file=$1 key=$2
    [ -r "$file" ] || return 1
    awk -v want="$key" '
        { line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
        line == "" || substr(line, 1, 1) == "#" { next }
        {
            eq = index(line, "=")
            if (eq == 0) { print "not an assignment: " line > "/dev/stderr"; exit 2 }
            name = substr(line, 1, eq - 1)
            value = substr(line, eq + 1)
            sub(/^[ \t]+/, "", value); sub(/[ \t]+$/, "", value)
            if (name == want) {
                n = length(value)
                if (n >= 2 && (substr(value,1,1) == "\"" || substr(value,1,1) == "\047") \
                    && substr(value,1,1) == substr(value,n,1)) {
                    value = substr(value, 2, n - 2)
                }
                print value
                found = 1
            }
        }
        END { exit(found ? 0 : 1) }
    ' "$file"
}

# The machine file's value, unless the calling shell exported an override.
# `$3` names the variable for the override message; `$4` is the fallback.
_setting() {
    local file=$1 key=$2 from_file override
    from_file=$(_env_file_lookup "$file" "$key" 2>/dev/null) || from_file=''
    override=${!key-}
    if [ -n "$override" ] && [ "$override" != "$from_file" ]; then
        _setup_error "the exported $key overrides $file for this run"
        printf '%s' "$override"
        return
    fi
    if [ -n "$from_file" ]; then
        printf '%s' "$from_file"
        return
    fi
    printf '%s' "$override"
}

# Print the interpreter path for an environment directory.
_conda_python() {
    printf '%s/bin/python3\n' "$1"
}

_require_plain_name() {
    case "$2" in
        ''|.|..|*/*) _setup_die "$1 must be one plain name: $2" ;;
    esac
}

_require_absolute() {
    case "$2" in
        /*) ;;
        *) _setup_die "$1 must be an absolute path: $2" ;;
    esac
}

main() {
    local env_file=${TILELANG_HIVM_ENV_FILE:-}
    if [ -z "$env_file" ]; then
        local here
        here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        env_file=$here/../env.sh
    fi
    _require_absolute TILELANG_HIVM_ENV_FILE "$env_file"
    if [ ! -r "$env_file" ]; then
        _setup_die "cannot read the machine file $env_file; copy env.example.sh to env.sh and set TILELANG_HIVM_WHEEL_DIR"
    fi
    printf 'tilelang-hivm-setup-python: machine file %s\n' "$env_file" >&2

    local wheel_dir pip_index python_version source_root
    local conda_sh conda_env conda_envs_path cann_setup
    wheel_dir=$(_setting "$env_file" TILELANG_HIVM_WHEEL_DIR)
    pip_index=$(_setting "$env_file" TILELANG_HIVM_PIP_INDEX)
    python_version=$(_setting "$env_file" TILELANG_HIVM_PYTHON_VERSION)
    source_root=$(_setting "$env_file" TILELANG_HIVM_SOURCE_ROOT)
    conda_sh=$(_setting "$env_file" TILELANG_HIVM_CONDA_SH)
    conda_env=$(_setting "$env_file" TILELANG_HIVM_CONDA_ENV)
    conda_envs_path=$(_setting "$env_file" TILELANG_HIVM_CONDA_ENVS_PATH)
    cann_setup=$(_setting "$env_file" TILELANG_HIVM_CANN_SETUP)

    # `PIP_INDEX` has no default in the file because an empty value means "install
    # from no index at all", which is a legitimate offline run. Only an unset
    # value takes the mirror.
    if [ -z "$pip_index" ] && [ -z "${TILELANG_HIVM_PIP_INDEX-}" ] \
       && ! grep -q '^[[:space:]]*TILELANG_HIVM_PIP_INDEX=' "$env_file" 2>/dev/null; then
        pip_index='https://mirrors.aliyun.com/pypi/simple/'
    fi
    [ -n "$python_version" ] || python_version=3.11
    [ -n "$conda_env" ] || conda_env=mk
    [ -n "$source_root" ] || source_root=$PWD

    _require_plain_name TILELANG_HIVM_CONDA_ENV "$conda_env"
    [ -n "$wheel_dir" ] || _setup_die 'TILELANG_HIVM_WHEEL_DIR is required: set it in '"$env_file"' to the directory with the torch and torch_npu wheels'
    _require_absolute TILELANG_HIVM_WHEEL_DIR "$wheel_dir"
    [ -d "$wheel_dir" ] || _setup_die "no such wheel directory: $wheel_dir"
    if [ -n "$conda_envs_path" ]; then
        _require_absolute TILELANG_HIVM_CONDA_ENVS_PATH "$conda_envs_path"
    fi

    # Report where the environment will live before any download starts, so a
    # wrong directory is visible in the first line of a long run.
    printf 'tilelang-hivm-setup-python: environment %s under %s\n' \
           "$conda_env" "${conda_envs_path:-<installation prefix>/envs}" >&2

    local torch_wheel torch_npu_wheel triton_wheel
    torch_wheel=$(find "$wheel_dir" -maxdepth 1 -name 'torch-*.whl' ! -name 'torch_npu-*' | sort | tail -1)
    torch_npu_wheel=$(find "$wheel_dir" -maxdepth 1 -name 'torch_npu-*.whl' | sort | tail -1)
    triton_wheel=$(find "$wheel_dir" -maxdepth 1 -name 'triton_ascend-*.whl' | sort | tail -1)
    [ -n "$torch_wheel" ] || _setup_die "no torch-*.whl in $wheel_dir"
    [ -n "$torch_npu_wheel" ] || _setup_die "no torch_npu-*.whl in $wheel_dir"
    case "$(basename "$torch_wheel")" in
        *+cpu*) ;;
        *) _setup_die "the torch wheel is not a +cpu build: $(basename "$torch_wheel")" ;;
    esac

    # The installation named by the file is reused. Nothing is installed here:
    # the machine file points at the installation the Gates source, so a second
    # Conda would be one the file does not describe.
    [ -n "$conda_sh" ] || _setup_die 'TILELANG_HIVM_CONDA_SH is required: set it in '"$env_file"' to the conda.sh of the installation to use'
    [ -r "$conda_sh" ] || _setup_die "TILELANG_HIVM_CONDA_SH is not readable: $conda_sh"
    # `conda.sh` normally sits at `<prefix>/etc/profile.d/conda.sh`, so the
    # prefix is two levels above its directory. The layout is checked by
    # requiring `<prefix>/bin/conda` to exist rather than trusting the depth
    # alone, so an unusual layout fails with a named error instead of silently
    # provisioning into the wrong directory.
    local conda conda_base
    conda_base=$(cd "$(dirname "$conda_sh")/../.." 2>/dev/null && pwd) || conda_base=''
    [ -n "$conda_base" ] || _setup_die "cannot locate the Conda prefix from $conda_sh"
    conda=$conda_base/bin/conda
    [ -x "$conda" ] || _setup_die "no conda executable at $conda (from TILELANG_HIVM_CONDA_SH=$conda_sh)"

    # The environment directory, in the same order Conda searches: the configured
    # directory first, then the installation's own prefix.
    local env_dir=''
    if [ -n "$conda_envs_path" ] && [ -d "$conda_envs_path/$conda_env" ]; then
        env_dir=$conda_envs_path/$conda_env
    elif [ -d "$conda_base/envs/$conda_env" ]; then
        env_dir=$conda_base/envs/$conda_env
    fi

    if [ -n "$env_dir" ]; then
        printf 'tilelang-hivm-setup-python: reusing environment %s at %s\n' \
               "$conda_env" "$env_dir" >&2
    else
        printf 'tilelang-hivm-setup-python: creating environment %s (python %s) in %s\n' \
               "$conda_env" "$python_version" "${conda_envs_path:-$conda_base/envs}" >&2
        # conda-forge is named explicitly: the default channel needs an accepted
        # Terms of Service on a fresh installation, and an interactive prompt
        # inside a provisioning script is a hang, not a failure.
        local create_args=(-y -q -n "$conda_env" -c conda-forge --override-channels
                           "python=$python_version")
        if [ -n "$conda_envs_path" ]; then
            create_args=(-p "$conda_envs_path/$conda_env" -c conda-forge --override-channels
                         "python=$python_version")
        fi
        "$conda" create "${create_args[@]}" >&2 \
            || _setup_die "cannot create the conda environment $conda_env"
        env_dir=${conda_envs_path:-$conda_base/envs}/$conda_env
    fi

    local python
    python=$(_conda_python "$env_dir")
    [ -x "$python" ] || _setup_die "no interpreter at $python"

    local index_args=()
    if [ -n "$pip_index" ]; then
        index_args=(--index-url "$pip_index")
    fi

    # The three vendor wheels install with --no-deps: their dependency metadata
    # names the CUDA companion packages, which this machine cannot use. The
    # dependencies that are actually needed are named below instead.
    printf 'tilelang-hivm-setup-python: installing the vendor wheels\n' >&2
    "$python" -m pip install -q --no-deps \
        "$torch_wheel" "$torch_npu_wheel" ${triton_wheel:+"$triton_wheel"} >&2 \
        || _setup_die 'cannot install the vendor wheels'

    # `apache-tvm-ffi` is pinned to the version the TileLang source expects. A
    # newer one re-registers `ffi.Tensor` and the editable build fails with
    # "already has a registered class"; `z3-solver` is required by the configure
    # step, which refuses to run without it.
    printf 'tilelang-hivm-setup-python: installing the Python dependencies\n' >&2
    "$python" -m pip install -q "${index_args[@]}" \
        filelock typing-extensions sympy networkx jinja2 numpy pyyaml psutil \
        cloudpickle ml_dtypes attrs decorator tqdm fsspec Cython pybind11 \
        cmake ninja patchelf scikit-build-core 'apache-tvm-ffi==0.1.12' \
        'z3-solver>=4.13.0' pytest >&2 \
        || _setup_die 'cannot install the Python dependencies'

    if [ -r "$source_root/CMakeLists.txt" ]; then
        printf 'tilelang-hivm-setup-python: installing TileLang editable from %s\n' "$source_root" >&2
        # The checkout usually belongs to the container user, and this script is
        # normally run as root because installing into a Conda prefix needs it.
        # Configure writes into `<source>/build`, so without this the build tree
        # ends up owned by root and the next build Gate, which runs as the
        # container user, fails with `ninja: error: rebuilding 'build.ninja':
        # Error writing to build log: Permission denied`. Record the owner first,
        # restore it after, and skip both when this already runs as that owner.
        local owner=''
        if [ "$(id -u)" = 0 ] && [ -d "$source_root" ]; then
            owner=$(stat -c '%u:%g' "$source_root" 2>/dev/null || true)
        fi

        # `USE_ASCEND=ON USE_CUDA=OFF` is not optional. Left to autodetection the
        # configure step picks the CUDA backend on a machine that has no CUDA
        # device, and the resulting library fails only later, inside a kernel.
        ( cd "$source_root" && \
          SKBUILD_CMAKE_DEFINE='USE_ASCEND=ON;USE_CUDA=OFF' \
          "$python" -m pip install -q --no-deps --no-build-isolation -e . >&2 ) \
            || _setup_die 'cannot install TileLang editable'

        if [ -n "$owner" ] && [ "$owner" != "0:0" ]; then
            # Only the directories this step can have created: the build trees
            # and the egg-info. Touching the whole checkout would be a surprise.
            local created
            for created in "$source_root/build" "$source_root"/build-* \
                           "$source_root"/*.egg-info; do
                [ -e "$created" ] || continue
                chown -R "$owner" "$created" 2>/dev/null || true
            done
        fi
    fi

    # Prove the environment instead of describing it: torch_npu refuses to load
    # without the CANN layout on the loader path, and a machine without a device
    # reports is_available() false rather than raising, so both are checked.
    printf 'tilelang-hivm-setup-python: verifying the environment\n' >&2
    if [ -n "$cann_setup" ]; then
        [ -r "$cann_setup" ] || _setup_die "TILELANG_HIVM_CANN_SETUP is not readable: $cann_setup"
        # shellcheck disable=SC1090  # the path is a validated absolute path
        . "$cann_setup" >&2 || _setup_die "cannot apply CANN setup: $cann_setup"
    else
        _setup_error 'TILELANG_HIVM_CANN_SETUP is unset; the NPU check below will fail without it'
    fi
    "$python" - <<'TILELANG_HIVM_SETUP_VERIFY' >&2 || _setup_die 'the environment does not reach the NPU'
import sys
import tilelang
import torch
import torch_npu

if not torch.npu.is_available():
    print('torch.npu.is_available() is false', file=sys.stderr)
    raise SystemExit(1)
print('tilelang %s, torch %s, torch_npu %s, npu available'
      % (tilelang.__version__, torch.__version__, torch_npu.__version__))
TILELANG_HIVM_SETUP_VERIFY

    printf 'tilelang-hivm-setup-python: ready\n' >&2
    # Report the two values the Gates need, so they can be copied into the
    # machine file. They are not required there to run this script: the script
    # derives them, and a first run is exactly when they are still unknown.
    printf 'TILELANG_HIVM_CONDA_SH=%s\n' "$conda_sh"
    printf 'TILELANG_HIVM_CONDA_ENV=%s\n' "$conda_env"
    if [ -n "$conda_envs_path" ]; then
        printf 'TILELANG_HIVM_CONDA_ENVS_PATH=%s\n' "$conda_envs_path"
    fi
}

main "$@"
