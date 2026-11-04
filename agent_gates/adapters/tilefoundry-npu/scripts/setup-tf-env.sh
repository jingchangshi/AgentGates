#!/usr/bin/env bash
# Provision the in-container TileFoundry environment.
#
# ONE environment now: a conda env under /home, containing everything. The
# earlier split into a 3.12 CPU venv and a 3.11 NPU conda env is gone, because
# torch_npu 2.12.0 exists as a cp312 build matching torch 2.12.0+cpu.
#
# WHY /home: it is bind-mounted, so this environment survives container
# recreation with no reinstall and no re-download. Rebuilding costs about an
# hour at direct network speed.
#
# NETWORK: direct access runs at ~58 KB/s; through the host proxy it is
# ~1.7 MB/s, about 30x. torch alone is 184 MB. The proxy exports are read from
# PROXY_SH and passed into docker exec with -e.
#
# IDEMPOTENT: a re-run verifies what exists and installs only what is missing.
#
# Exit codes: 0 provisioned, 2 configuration problem, 75 BLOCKED_ENV.
set -uo pipefail

PROG="setup-tf-env.sh"
log() { printf '%s: %s\n' "$PROG" "$*" >&2; }
die() { log "$*"; exit "${2:-2}"; }

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TF_ENV_FILE:-$HERE/../env.sh}"

declare -A CFG=()
if [ -r "$ENV_FILE" ]; then
    n=0
    while IFS= read -r line || [ -n "$line" ]; do
        n=$((n + 1))
        case "$line" in
            ''|'#'*) continue ;;
        esac
        case "$line" in
            *=*) ;;
            *) die "$ENV_FILE:$n: not NAME=value" ;;
        esac
        k="${line%%=*}"; v="${line#*=}"
        [ -n "${CFG[$k]+x}" ] && die "$ENV_FILE:$n: duplicate key: $k"
        CFG[$k]="$v"
    done < "$ENV_FILE"
fi

TF_CONDA_ENV="${TF_CONDA_ENV:-${CFG[TF_CONDA_ENV]:-/home/s00653124/.conda/envs/tilefoundry-npu}}"
CONDA_SH="${CONDA_SH:-${CFG[CONDA_SH]:-/data/miniconda3/etc/profile.d/conda.sh}}"
CONTAINER_NAME="${CONTAINER_NAME:-${CFG[CONTAINER_NAME]:-}}"
CONTAINER_USER="${CONTAINER_USER:-${CFG[CONTAINER_USER]:-s00653124}}"
PROXY_SH="${PROXY_SH:-${CFG[PROXY_SH]:-/home/s00653124/set_proxy.sh}}"
TORCH_WHEEL="${TORCH_WHEEL:-https://download-r2.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl}"
MIRROR="${MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"

if [ -e /.dockerenv ]; then
    RUN() { bash -c "$1"; }
else
    [ -n "$CONTAINER_NAME" ] || die "CONTAINER_NAME is required when provisioning from the host"
    docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || die "container not found: $CONTAINER_NAME" 75
    # The proxy values are resolved on the HOST and injected, because the
    # container inherits a different environment and would otherwise go direct.
    proxy_args=()
    if [ -r "$PROXY_SH" ]; then
        while IFS= read -r assignment; do
            proxy_args+=(-e "$assignment")
        done < <(bash -c ". $PROXY_SH >/dev/null 2>&1; printf '%s\n' \
            "http_proxy=$http_proxy" "https_proxy=$https_proxy" "no_proxy=$no_proxy"")
    else
        log "no proxy script at $PROXY_SH; installs will run at ~58 KB/s"
    fi
    RUN() { docker exec -u "$CONTAINER_USER" "${proxy_args[@]:-}" "$CONTAINER_NAME" bash -c "$1"; }
fi

log "environment: $TF_CONDA_ENV (python 3.12)"

# 1. The interpreter. repo.anaconda.com demands an interactive Terms-of-Service
#    acceptance; that is a legal agreement, not something this script clicks
#    through on the user behalf, so conda-forge is used instead.
RUN "test -x $TF_CONDA_ENV/bin/python || (export CONDA_ENVS_PATH=\$(dirname $TF_CONDA_ENV); . $CONDA_SH && conda create -p $TF_CONDA_ENV -y python=3.12 --override-channels -c conda-forge)" \
    || die "could not create the conda environment"
RUN "$TF_CONDA_ENV/bin/python --version" || die "the interpreter is not usable"

# 2. torch, from the exact cp312 wheel. numpy is pulled by torch.
RUN "$TF_CONDA_ENV/bin/python -m pip install --no-cache-dir '$TORCH_WHEEL'" \
    || die "torch install failed"

# 3. torch_npu 2.12.0 is the cp312 build matching torch 2.12.0. The older vendor
#    wheels are cp311-only and pair with torch 2.10, so they are not used here.
RUN "$TF_CONDA_ENV/bin/python -m pip install --no-cache-dir torch_npu==2.12.0 --index-url $MIRROR" \
    || die "torch_npu install failed"

# 4. The TileFoundry dependencies, and z3, which the acceptance condition needs.
RUN "$TF_CONDA_ENV/bin/python -m pip install --no-cache-dir 'isl-python==0.1.11' 'jinja2==3.1.6' 'graphviz==0.21' 'apache-tvm-ffi==0.1.15.post2' 'ortools==9.15.6755' z3-solver numpy --index-url $MIRROR" \
    || die "TileFoundry dependency install failed"

# 5. Verify. z3 is checked by SOLVING, because an import only proves unpacking.
RUN "$TF_CONDA_ENV/bin/python -c 'import z3; x=z3.Int(\"x\"); s=z3.Solver(); s.add(x>3, x<6); assert str(s.check())==\"sat\"; print(\"z3 verified by solving\")'" \
    || die "z3 is installed but does not solve" 75

log "environment ready: $TF_CONDA_ENV/bin/python"
log "CANN must be sourced before importing torch_npu: see CANN_SET_ENV in env.example.sh"
exit 0
