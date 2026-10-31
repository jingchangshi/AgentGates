#!/usr/bin/env bash
# Create the Python environment the tla-npu Gate runs in.
#
# This file is the only place the Python dependency set is written. Run it once
# inside the container as the container user. Root is never needed: the Conda
# installation it uses already exists and is writable by that user, and nothing
# here installs system packages.
#
#     bash <checkout>/adapters/tla-npu/scripts/setup-python-env.sh
#
# The configuration is the machine file adapters/tla-npu/env.sh, the same file
# the Gate reads. Start from the committed template:
#
#     cp adapters/tla-npu/env.example.sh adapters/tla-npu/env.sh
#     chmod 600 adapters/tla-npu/env.sh
#
# Then set the values this script needs:
#
#     TLA_NPU_WHEEL_DIR          directory holding the vendor wheels; required
#     TLA_NPU_CONDA_SH           conda.sh of the installation to use; required
#     TLA_NPU_CONDA_ENV          environment to create or reuse
#     TLA_NPU_CONDA_ENVS_PATH    extra directory Conda searches for environments
#     TLA_NPU_PYTHON_VERSION     interpreter version; default 3.11
#     TLA_NPU_PIP_INDEX          index for the small dependencies
#     TLA_NPU_CANN_SETUP         CANN set_env.sh, needed by the final check
#
# The file is the primary mechanism because a shell export leaves no record.
# One exception, narrow: TLA_NPU_ENV_FILE selects which file to read. A bootstrap
# value cannot come from the file it names.
#
# The Conda installation is reused, never created. TLA_NPU_CONDA_SH names the
# installation the Gate sources, so the environment this script builds is the
# environment the Gate runs in.
#
# The exit status is 0 only when the environment imports the TLA namespace and
# reports an available NPU. Anything else is printed on stderr and returns
# non-zero.
#
# Why the wheels are local files rather than an index download: this is an Ascend
# server. pip install torch from PyPI resolves to the CUDA build and pulls
# several gigabytes of nvidia-* wheels that cannot run here. The vendor directory
# ships torch-<v>+cpu and torch_npu-<v>, which are the pair the Gate is verified
# against. The triton_ascend wheel is a vendor build too: it is not on any public
# index. Small pure-Python dependencies still come from an index, which is why
# this script needs network access once.

set -o pipefail

_setup_error() {
    printf 'tla-npu-setup-python: %s\n' "$*" >&2
}

_setup_die() {
    _setup_error "$@"
    exit 1
}

# The strict NAME=value grammar the Gate uses. A machine file is data, so reading
# it never executes Host code. Kept in step with _parse_env_file in
# scripts/tla_gate.py; a value that the Gate rejects must not be accepted here.
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

# The machine file's value, or the fallback when the file leaves the key out.
_setting() {
    local file=$1 key=$2 fallback=$3 value
    value=$(_env_file_lookup "$file" "$key" 2>/dev/null) || value=''
    if [ -n "$value" ]; then
        printf '%s' "$value"
        return
    fi
    printf '%s' "$fallback"
}

# The machine file to read: the selector wins, then the checkout's own file.
_env_file() {
    local explicit=${TLA_NPU_ENV_FILE:-}
    if [ -n "$explicit" ]; then
        case "$explicit" in
            /*) ;;
            *) _setup_die "TLA_NPU_ENV_FILE must be an absolute path: $explicit" ;;
        esac
        [ -f "$explicit" ] || _setup_die "TLA_NPU_ENV_FILE does not exist: $explicit"
        printf '%s' "$explicit"
        return
    fi
    local here
    here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    if [ -f "$here/env.sh" ]; then
        printf '%s' "$here/env.sh"
        return
    fi
    printf ''
}

# Newest wheel matching one pattern anywhere under a directory.
# Sorting by the full path would order by package directory name rather than by
# version, so the candidate list is sorted and the last (newest by name) wins
# only among wheels in the same directory; across directories the newest mtime
# wins, which is what "the version somebody put there most recently" means.
_newest_wheel() {
    local directory pattern
    directory=$1
    pattern=$2
    find "$directory" -name "$pattern" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

main() {
    local env_file wheel_dir conda_sh conda_env conda_envs_path python_version
    local pip_index cann python env_dir

    env_file=$(_env_file)
    [ -n "$env_file" ] || _setup_die 'no machine file found; copy env.example.sh to env.sh and configure it'

    wheel_dir=$(_setting "$env_file" TLA_NPU_WHEEL_DIR '')
    conda_sh=$(_setting "$env_file" TLA_NPU_CONDA_SH '')
    conda_env=$(_setting "$env_file" TLA_NPU_CONDA_ENV '')
    conda_envs_path=$(_setting "$env_file" TLA_NPU_CONDA_ENVS_PATH '')
    python_version=$(_setting "$env_file" TLA_NPU_PYTHON_VERSION '3.11')
    pip_index=$(_setting "$env_file" TLA_NPU_PIP_INDEX 'https://mirrors.aliyun.com/pypi/simple/')
    cann=$(_setting "$env_file" TLA_NPU_CANN_SETUP '')

    [ -n "$wheel_dir" ] || _setup_die 'TLA_NPU_WHEEL_DIR is required; it names the directory holding the vendor wheels'
    [ -n "$conda_sh" ] || _setup_die 'TLA_NPU_CONDA_SH is required; it names the Conda installation to use'
    [ -n "$conda_env" ] || _setup_die 'TLA_NPU_CONDA_ENV is required; it names the environment to create'
    [ -r "$conda_sh" ] || _setup_die "conda.sh not readable: $conda_sh"
    [ -d "$wheel_dir" ] || _setup_die "wheel directory does not exist: $wheel_dir"

    # The vendor wheels this environment is built from. Selected by explicit
    # pattern rather than globbed blindly, so a directory holding two versions
    # cannot silently provision the one nobody chose. The search is recursive
    # because the wheels are kept in per-package subdirectories
    # (Packages/torch, Packages/tla-...), and a flat-only find would report
    # "no wheel" on a machine where the wheels are right there one level down.
    local triton_wheel torch_wheel npu_wheel
    # An explicit wheel wins over discovery. This is the only way to pin the
    # exact build an operator validated, and the TLA API changed between vendor
    # builds, so "newest by mtime" is not a safe default on its own.
    local triton_override
    triton_override=$(_setting "$env_file" TLA_NPU_TRITON_WHEEL '')
    if [ -n "$triton_override" ]; then
        [ -f "$triton_override" ] || _setup_die "TLA_NPU_TRITON_WHEEL is not a file: $triton_override"
        triton_wheel=$triton_override
    else
        triton_wheel=$(_newest_wheel "$wheel_dir" 'triton_ascend-*-cp311-*.whl')
    fi
    torch_wheel=$(_newest_wheel "$wheel_dir" 'torch-*+cpu-cp311-*.whl')
    npu_wheel=$(_newest_wheel "$wheel_dir" 'torch_npu-*-cp311-*.whl')
    [ -n "$triton_wheel" ] || _setup_die "no triton_ascend cp311 wheel under $wheel_dir"
    [ -f "$triton_wheel" ] || _setup_die "triton wheel is not a file: $triton_wheel"

    # CANN first: conda activation and torch_npu's later import both resolve
    # native libraries from the CANN layout this puts on the loader path.
    if [ -n "$cann" ]; then
        [ -r "$cann" ] || _setup_die "CANN setup not readable: $cann"
        # shellcheck disable=SC1090
        . "$cann" || _setup_die "sourcing CANN setup failed: $cann"
        [ -n "${ASCEND_HOME_PATH:-}" ] || _setup_die "CANN setup did not define ASCEND_HOME_PATH: $cann"
    fi

    if [ -n "$conda_envs_path" ]; then
        export CONDA_ENVS_PATH="$conda_envs_path"
    fi
    # shellcheck disable=SC1090
    . "$conda_sh" || _setup_die "sourcing conda.sh failed: $conda_sh"

    env_dir=$(conda env list --json 2>/dev/null | python3 -c '
import json, sys
name = sys.argv[1]
for prefix in json.load(sys.stdin).get("envs", []):
    if prefix.rstrip("/").rsplit("/", 1)[-1] == name:
        print(prefix)
        break
' "$conda_env")

    if [ -z "$env_dir" ]; then
        _setup_error "creating environment $conda_env (python $python_version)"
        conda create -y -n "$conda_env" "python=$python_version" \
            || _setup_die "conda create $conda_env failed"
        env_dir=$(conda env list --json 2>/dev/null | python3 -c '
import json, sys
name = sys.argv[1]
for prefix in json.load(sys.stdin).get("envs", []):
    if prefix.rstrip("/").rsplit("/", 1)[-1] == name:
        print(prefix)
        break
' "$conda_env")
    else
        _setup_error "reusing environment $conda_env at $env_dir"
    fi

    [ -n "$env_dir" ] || _setup_die "cannot locate the environment directory for $conda_env"
    python="$env_dir/bin/python3"
    [ -x "$python" ] || _setup_die "interpreter not found after environment creation: $python"

    # torch first, then torch_npu: torch_npu's metadata pins the matching torch
    # build, and installing it in the other order lets pip pull a CUDA torch.
    local -a pip_args=(--disable-pip-version-check --no-cache-dir)
    if [ -n "$pip_index" ]; then
        pip_args+=(-i "$pip_index")
        pip_args+=(--trusted-host "$(printf '%s' "$pip_index" | sed -e 's|^https\?://||' -e 's|/.*$||')")
    fi

    if [ -n "$torch_wheel" ]; then
        [ -f "$torch_wheel" ] || _setup_die "torch wheel is not a file: $torch_wheel"
        _setup_error "installing $torch_wheel"
        "$python" -m pip install "${pip_args[@]}" "$torch_wheel" \
            || _setup_die "installing torch failed"
    else
        _setup_error 'no vendor torch wheel found; relying on the environment for torch'
    fi

    if [ -n "$npu_wheel" ]; then
        [ -f "$npu_wheel" ] || _setup_die "torch_npu wheel is not a file: $npu_wheel"
        _setup_error "installing $npu_wheel"
        "$python" -m pip install "${pip_args[@]}" "$npu_wheel" \
            || _setup_die "installing torch_npu failed"
    fi

    # The vendor Triton build last, so it overlays the community triton wheel its
    # metadata declares. Dependencies are resolved, not skipped: the wheel pins
    # exact versions of attrs, numpy, scipy, decorator, psutil, pytest and
    # pytest-xdist, and an environment with different ones is not the environment
    # the Gate was verified against.
    _setup_error "installing $triton_wheel"
    "$python" -m pip install "${pip_args[@]}" "$triton_wheel" \
        || _setup_die "installing triton_ascend failed"

    # The final check is the one that matters: the TLA namespace must import and
    # the device must be visible. A wheel that installs but cannot be imported is
    # not provisioned.
    _setup_error 'verifying the environment'
    "$python" - <<'TLA_NPU_CHECK' || _setup_die 'environment verification failed'
import sys

try:
    import torch
    import torch_npu
    import triton
    import triton.language as tl
    import triton.language.extra.tla as tla
    import triton.backends.ascend as ascend
except BaseException as error:
    print("import failed: %s: %s" % (type(error).__name__, error), file=sys.stderr)
    raise SystemExit(1)

print("triton", triton.__version__)
print("triton_ascend", ascend.__version__)
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("tla_namespace", sorted(name for name in dir(tla) if not name.startswith("_"))[:12])
# A wheel that ships the tla package but not its entry points cannot run the
# Gate test, and installing it would produce an environment that looks correct
# until the test fails on an attribute error. Refuse it here instead.
for attribute in ("jit", "alloc_buffer", "view", "copy", "simd"):
    if not hasattr(tla, attribute):
        print("tla is missing %r; this triton_ascend build cannot run the Gate test"
              % attribute, file=sys.stderr)
        raise SystemExit(1)
for attribute in ("simd", "cube"):
    if not hasattr(tla.scope, attribute):
        print("tla.scope is missing %r; this triton_ascend build cannot run the Gate test"
              % attribute, file=sys.stderr)
        raise SystemExit(1)
# The Python entry points come from the tla package; the operations they call
# live in the compiled extension. A wheel can ship one without the other - the
# post20261030 build does exactly that - and the failure then appears only when
# a kernel is compiled. Probe the native side too, so provisioning refuses the
# wheel instead of leaving a time bomb.
try:
    from triton._C.libtriton import ir as _ir
    from triton._C.libtriton.ascend import ir as _air
    _ctx = _ir.context()
    # The TLA operations live on the dedicated tla_builder, which the compiler
    # mounts onto the code generator before visiting a kernel. The generic
    # ascendnpu_ir_builder and the plain ir.builder do NOT carry them, so probing
    # either would reject a good wheel.
    _builder = _air.tla_builder(_ctx, 'Ascend950PR_9579')
except BaseException as error:
    print("cannot construct the Ascend IR builder: %s: %s" % (type(error).__name__, error),
          file=sys.stderr)
    raise SystemExit(1)
required_native = ('create_tla_alloc', 'create_tla_buffer_type', 'create_tla_bind',
                   'create_tla_subview', 'create_tla_reinterpret_cast',
                   'create_hivm_load', 'create_hivm_store', 'create_simd_vlds',
                   'create_simd_vsts', 'create_simd_binary', 'create_simd_vreduce',
                   'create_scope_op')
missing_native = [n for n in required_native if not hasattr(_builder, n)]
if missing_native:
    print("this triton_ascend build is missing %d native TLA builder method(s): %s"
          % (len(missing_native), ", ".join(missing_native)), file=sys.stderr)
    print("its tla Python package cannot compile a kernel; install a build that has them",
          file=sys.stderr)
    raise SystemExit(1)
print("tla_native_api", "complete")
print("tla_api", "complete")
if not torch.npu.is_available():
    print("torch.npu.is_available() is False", file=sys.stderr)
    raise SystemExit(1)
print("npu_available", True)
TLA_NPU_CHECK

    _setup_error "environment $conda_env is ready at $env_dir"
    return 0
}

main "$@"