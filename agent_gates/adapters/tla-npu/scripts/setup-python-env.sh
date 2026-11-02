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
# Configuration is the machine file adapters/tla-npu/env.sh, the same file the
# Gate reads. Start from the committed template:
#
#     cp adapters/tla-npu/env.example.sh adapters/tla-npu/env.sh
#     chmod 600 adapters/tla-npu/env.sh
#
# Keys this script reads, and what each one controls:
#
#   TLA_NPU_CONDA_SH         conda.sh of the installation to reuse   (required)
#   TLA_NPU_CONDA_ENV        environment to create or reuse          (required)
#   TLA_NPU_WHEEL_DIR        directory holding the vendor wheels     (required)
#   TLA_NPU_CANN_SETUP       CANN set_env.sh, sourced before any import
#   TLA_NPU_CONDA_ENVS_PATH  extra directory Conda searches for environments
#   TLA_NPU_PYTHON_VERSION   interpreter version, default 3.11
#   TLA_NPU_PIP_INDEX        index for the small pure-Python dependencies
#   TLA_NPU_TRITON_WHEEL     exact triton_ascend wheel to install; overrides the
#                            newest-by-mtime search in TLA_NPU_WHEEL_DIR
#
# TLA_NPU_ENV_FILE is the single exception to "the file is the only source": it
# selects which file to read, and a bootstrap value cannot come from the file it
# names.
#
# The Conda installation is reused, never created. TLA_NPU_CONDA_SH names the
# installation the Gate sources, so the environment built here is the environment
# the Gate runs in.
#
# Exit status is 0 only when the environment imports the TLA namespace, carries
# the native builder methods a kernel needs, and reports an available NPU.
#
# Why the wheels are local files rather than index downloads: this is an Ascend
# server. pip install torch from PyPI resolves to the CUDA build and pulls
# several gigabytes of nvidia-* wheels that cannot run here. The vendor directory
# ships torch-<v>+cpu and torch_npu-<v>, which are the pair the Gate is verified
# against. The triton_ascend wheel is a vendor build too: it is not on any public
# index. Small pure-Python dependencies still come from an index, which is why
# this script needs network access once.
#
# Reading and installing are separated on purpose: every value is resolved and
# validated before anything is downloaded, so a typo in the machine file cannot
# leave the environment half-built.

set -o pipefail

_setup_error() {
    printf 'tla-npu-setup-python: %s\n' "$*" >&2
}

_setup_die() {
    _setup_error "$@"
    exit 1
}

# The strict NAME=value grammar the Gate uses. A machine file is data, so reading
# it never executes host code. Kept in step with _parse_env_file in
# scripts/tla_gate.py: a value the Gate rejects must not be accepted here.
#
# A malformed line is reported on stderr and makes the lookup fail with status 2,
# which is distinct from status 1 ("this file does not set that key"). Callers
# must not conflate the two: a file with a typo has to stop provisioning, not
# silently fall back to a default that hides it.
_env_file_lookup() {
    local file=$1 key=$2
    [ -r "$file" ] || return 1
    awk -v want="$key" '
        { line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
        line == "" || substr(line, 1, 1) == "#" { next }
        {
            eq = index(line, "=")
            # Set a flag rather than calling exit here. An exit in the main
            # block still runs the END action, and the status from END wins, so
            # the 2 below would never reach the caller. That is exactly how the
            # malformed-file check came to be dead code, so keep the flag.
            if (eq == 0) { print "not an assignment: " line > "/dev/stderr"; bad = 1; next }
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
        END { exit(bad ? 2 : (found ? 0 : 1)) }
    ' "$file"
}

# One machine-file value, or the fallback when the file does not set the key. A
# malformed file is fatal here rather than silently defaulted: the previous
# version discarded stderr and so turned the parser's "this file is broken" into
# "use the default", which is how a typo could provision an environment the
# operator never configured.
_env_value() {
    local file=$1 key=$2 fallback=$3 value rc detail
    value=$(_env_file_lookup "$file" "$key" 2>/dev/null)
    rc=$?
    if [ "$rc" -eq 2 ]; then
        detail=$(_env_file_lookup "$file" "$key" 2>&1 >/dev/null)
        _setup_die "$file is malformed: $detail"
    fi
    if [ "$rc" -eq 0 ] && [ -n "$value" ]; then
        printf '%s' "$value"
        return 0
    fi
    printf '%s' "$fallback"
}

# The machine file to read: the selector wins, then the checkout's own file.
_env_file() {
    local explicit=${TLA_NPU_ENV_FILE:-} here
    if [ -n "$explicit" ]; then
        case "$explicit" in
            /*) ;;
            *) _setup_die "TLA_NPU_ENV_FILE must be an absolute path: $explicit" ;;
        esac
        [ -f "$explicit" ] || _setup_die "TLA_NPU_ENV_FILE does not exist: $explicit"
        printf '%s' "$explicit"
        return 0
    fi
    here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    if [ -f "$here/env.sh" ]; then
        printf '%s' "$here/env.sh"
        return 0
    fi
    printf ''
}

# Newest wheel matching one pattern anywhere under a directory. Sorting by full
# path would order by package directory name rather than by version, so the
# candidates are ordered by mtime: "the version somebody put there most recently".
_newest_wheel() {
    local directory=$1 pattern=$2
    find "$directory" -name "$pattern" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

# The Conda prefix for a named environment, or nothing when it is not created
# yet. Called before and after creation, so it exists once rather than being
# pasted inline at each call site.
_conda_env_dir() {
    local name=$1
    conda env list --json 2>/dev/null | python3 -c '
import json, sys
name = sys.argv[1]
for prefix in json.load(sys.stdin).get("envs", []):
    if prefix.rstrip("/").rsplit("/", 1)[-1] == name:
        print(prefix)
        break
' "$name"
}

# Resolve the machine file into CFG_* globals. Nothing is installed here.
_config_load() {
    CFG_FILE=$(_env_file)
    [ -n "$CFG_FILE" ] \
        || _setup_die 'no machine file found; copy env.example.sh to env.sh and configure it'

    CFG_WHEEL_DIR=$(_env_value "$CFG_FILE" TLA_NPU_WHEEL_DIR '')
    CFG_CONDA_SH=$(_env_value "$CFG_FILE" TLA_NPU_CONDA_SH '')
    CFG_CONDA_ENV=$(_env_value "$CFG_FILE" TLA_NPU_CONDA_ENV '')
    CFG_CONDA_ENVS_PATH=$(_env_value "$CFG_FILE" TLA_NPU_CONDA_ENVS_PATH '')
    CFG_PYTHON_VERSION=$(_env_value "$CFG_FILE" TLA_NPU_PYTHON_VERSION '3.11')
    CFG_PIP_INDEX=$(_env_value "$CFG_FILE" TLA_NPU_PIP_INDEX 'https://mirrors.aliyun.com/pypi/simple/')
    CFG_CANN=$(_env_value "$CFG_FILE" TLA_NPU_CANN_SETUP '')
    CFG_TRITON_WHEEL=$(_env_value "$CFG_FILE" TLA_NPU_TRITON_WHEEL '')

    [ -n "$CFG_WHEEL_DIR" ] \
        || _setup_die 'TLA_NPU_WHEEL_DIR is required; it names the directory holding the vendor wheels'
    [ -n "$CFG_CONDA_SH" ] \
        || _setup_die 'TLA_NPU_CONDA_SH is required; it names the Conda installation to use'
    [ -n "$CFG_CONDA_ENV" ] \
        || _setup_die 'TLA_NPU_CONDA_ENV is required; it names the environment to create'
    [ -r "$CFG_CONDA_SH" ] || _setup_die "conda.sh not readable: $CFG_CONDA_SH"
    [ -d "$CFG_WHEEL_DIR" ] || _setup_die "wheel directory does not exist: $CFG_WHEEL_DIR"
    if [ -n "$CFG_CANN" ] && [ ! -r "$CFG_CANN" ]; then
        _setup_die "CANN setup not readable: $CFG_CANN"
    fi
    if [ -n "$CFG_TRITON_WHEEL" ] && [ ! -f "$CFG_TRITON_WHEEL" ]; then
        _setup_die "TLA_NPU_TRITON_WHEEL is not a file: $CFG_TRITON_WHEEL"
    fi
    # The Gate rejects an environment name that is not a plain identifier, so
    # provisioning must reject it too. Building an environment the Gate will
    # then refuse is a failure discovered at the worst possible moment.
    case "$CFG_CONDA_ENV" in
        [A-Za-z0-9]* ) ;;
        * ) _setup_die "TLA_NPU_CONDA_ENV is not a plain environment name: $CFG_CONDA_ENV" ;;
    esac
    case "$CFG_CONDA_ENV" in
        *[!A-Za-z0-9._-]* ) _setup_die "TLA_NPU_CONDA_ENV is not a plain environment name: $CFG_CONDA_ENV" ;;
    esac
}

# Select the vendor wheels. The Triton build decides which TLA API and which
# native bindings exist, so an explicit TLA_NPU_TRITON_WHEEL wins over discovery:
# "newest by mtime" is not a safe default for a component whose API changes
# between vendor builds.
_wheels_select() {
    WHEEL_TRITON=${CFG_TRITON_WHEEL:-}
    if [ -z "$WHEEL_TRITON" ]; then
        WHEEL_TRITON=$(_newest_wheel "$CFG_WHEEL_DIR" 'triton_ascend-*-cp311-*.whl')
    fi
    WHEEL_TORCH=$(_newest_wheel "$CFG_WHEEL_DIR" 'torch-*+cpu-cp311-*.whl')
    WHEEL_NPU=$(_newest_wheel "$CFG_WHEEL_DIR" 'torch_npu-*-cp311-*.whl')
    [ -n "$WHEEL_TRITON" ] || _setup_die "no triton_ascend cp311 wheel under $CFG_WHEEL_DIR"
    [ -f "$WHEEL_TRITON" ] || _setup_die "triton wheel is not a file: $WHEEL_TRITON"
}

# Put CANN and Conda on the path, create or reuse the environment, and leave the
# interpreter in PYTHON and its prefix in ENV_DIR.
_env_prepare() {
    local env_dir
    # CANN first: conda activation and torch_npu's later import both resolve
    # native libraries from the layout this puts on the loader path.
    if [ -n "$CFG_CANN" ]; then
        # shellcheck disable=SC1090
        . "$CFG_CANN" || _setup_die "sourcing CANN setup failed: $CFG_CANN"
        [ -n "${ASCEND_HOME_PATH:-}" ] \
            || _setup_die "CANN setup did not define ASCEND_HOME_PATH: $CFG_CANN"
    fi

    if [ -n "$CFG_CONDA_ENVS_PATH" ]; then
        export CONDA_ENVS_PATH="$CFG_CONDA_ENVS_PATH"
    fi
    # shellcheck disable=SC1090
    . "$CFG_CONDA_SH" || _setup_die "sourcing conda.sh failed: $CFG_CONDA_SH"

    env_dir=$(_conda_env_dir "$CFG_CONDA_ENV")
    if [ -z "$env_dir" ]; then
        _setup_error "creating environment $CFG_CONDA_ENV (python $CFG_PYTHON_VERSION)"
        conda create -y -n "$CFG_CONDA_ENV" "python=$CFG_PYTHON_VERSION" \
            || _setup_die "conda create $CFG_CONDA_ENV failed"
        env_dir=$(_conda_env_dir "$CFG_CONDA_ENV")
    else
        _setup_error "reusing environment $CFG_CONDA_ENV at $env_dir"
    fi

    [ -n "$env_dir" ] || _setup_die "cannot locate the environment directory for $CFG_CONDA_ENV"
    ENV_DIR=$env_dir
    PYTHON=$env_dir/bin/python3
    [ -x "$PYTHON" ] || _setup_die "interpreter not found after environment creation: $PYTHON"
}

# pip options shared by every install: no cache (the wheels are local and large),
# and an index for the small pure-Python dependencies only.
_pip_install() {
    local -a args=(--disable-pip-version-check --no-cache-dir)
    if [ -n "$CFG_PIP_INDEX" ]; then
        args+=(-i "$CFG_PIP_INDEX")
        args+=(--trusted-host "$(printf '%s' "$CFG_PIP_INDEX" | sed -e 's|^https\?://||' -e 's|/.*$||')")
    fi
    "$PYTHON" -m pip install "${args[@]}" "$@" || _setup_die "installing $* failed"
}

# Install in dependency order. torch before torch_npu, because torch_npu's
# metadata pins the matching torch build and the other order lets pip resolve a
# CUDA torch. The vendor Triton build last, so it overlays the community triton
# wheel its metadata declares; its dependencies are resolved rather than skipped,
# because it pins exact versions and an environment with different ones is not
# the environment the Gate was verified against.
_wheels_install() {
    if [ -n "$WHEEL_TORCH" ]; then
        _setup_error "installing $WHEEL_TORCH"
        _pip_install "$WHEEL_TORCH"
    else
        _setup_error 'no vendor torch wheel found; relying on the environment for torch'
    fi
    if [ -n "$WHEEL_NPU" ]; then
        _setup_error "installing $WHEEL_NPU"
        _pip_install "$WHEEL_NPU"
    fi
    _setup_error "installing $WHEEL_TRITON"
    _pip_install "$WHEEL_TRITON"
}

# The check that decides the exit status: a wheel that installs but cannot be
# imported, or that imports without the native builder methods a kernel needs, is
# not provisioned. Each failure names its own cause.
_verify_environment() {
    _setup_error 'verifying the environment'
    "$PYTHON" - <<'TLA_NPU_CHECK' || _setup_die 'environment verification failed'
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
# The API surface the Gate test actually uses, plus the newer modules the
# triton-ascend main-dev line ships (dma, debug). A build that predates them can
# still pass the older checks, so naming them here is what makes provisioning
# refuse a wheel that cannot run current TLA code.
for attribute in ("jit", "alloc_buffer", "view", "copy", "simd", "dma"):
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
}

main() {
    _config_load
    _wheels_select
    _env_prepare
    _wheels_install
    _verify_environment
    _setup_error "environment $CFG_CONDA_ENV is ready at $ENV_DIR"
    return 0
}

main "$@"
