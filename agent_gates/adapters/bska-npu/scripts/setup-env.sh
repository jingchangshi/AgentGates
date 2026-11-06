#!/usr/bin/env bash
# One-time provisioning for the bska-npu Adapter (manual operator action).
#
# Installs node and claude into the running container, creates the conda
# environment under the configured envs directory, and installs helix
# (BiShengKernelAgent) into it. The Gate never calls this script.
#
#   bash scripts/setup-env.sh --base-env <validated-env-name> \
#       --helix-repo /home/<user>/workspace/BiShengKernelAgent
#
# --base-env names an existing conda environment that already imports torch,
# torch_npu and triton; it is cloned into BSKA_NPU_CONDA_ENV. Required only when
# the target environment does not exist yet.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER="$(cd "$HERE/.." && pwd)"
ENV_FILE="${BSKA_NPU_ENV_FILE:-$ADAPTER/env.sh}"
BASE_ENV=""
HELIX_REPO=""
PROXY="http://127.0.0.1:17890"
NODE_VERSION="v24.20.0"
NPM_REGISTRY="https://registry.npmmirror.com"
NODE_MIRROR="https://mirrors.aliyun.com/nodejs-release"
PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-env) BASE_ENV="$2"; shift 2 ;;
        --helix-repo) HELIX_REPO="$2"; shift 2 ;;
        --proxy) PROXY="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -r "$ENV_FILE" ] || { echo "machine file not readable: $ENV_FILE" >&2; exit 2; }
set -a; . "$ENV_FILE"; set +a

CONTAINER="${BSKA_NPU_CONTAINER_NAME:?BSKA_NPU_CONTAINER_NAME is required}"
ENVS_PATH="${BSKA_NPU_CONDA_ENVS_PATH:?}"
CONDA_ENV="${BSKA_NPU_CONDA_ENV:?}"
HELIX_REPO="${HELIX_REPO:-${BSKA_NPU_HELIX_REPO:-}}"
DST="$ENVS_PATH/$CONDA_ENV"

docker inspect "$CONTAINER" >/dev/null 2>&1 || { echo "container $CONTAINER does not exist" >&2; exit 2; }
[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" = true ] || { echo "container $CONTAINER is not running" >&2; exit 2; }

PROXY_ENV=( -e "http_proxy=$PROXY" -e "https_proxy=$PROXY" -e "HTTP_PROXY=$PROXY" -e "HTTPS_PROXY=$PROXY" -e "no_proxy=localhost,127.0.0.1,::1" -e "NO_PROXY=localhost,127.0.0.1,::1" )

echo "== node + claude =="
docker exec -i -u root "${PROXY_ENV[@]}" \
    -e "NODE_MIRROR=$NODE_MIRROR" -e "NODE_VERSION=$NODE_VERSION" -e "NPM_REGISTRY=$NPM_REGISTRY" \
    "$CONTAINER" bash -s <<'INNER'
set -e
if ! command -v node >/dev/null 2>&1; then
    cd /tmp
    curl -fsSL -m 300 -o node.tar.gz "$NODE_MIRROR/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.gz"
    mkdir -p /usr/local/lib/nodejs
    tar -xzf node.tar.gz -C /usr/local/lib/nodejs
    D=/usr/local/lib/nodejs/node-$NODE_VERSION-linux-x64
    ln -sf "$D/bin/node" /usr/local/bin/node
    ln -sf "$D/bin/npm" /usr/local/bin/npm
    ln -sf "$D/bin/npx" /usr/local/bin/npx
    rm -f node.tar.gz
fi
node --version
npm config set registry "$NPM_REGISTRY"
if ! command -v claude >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code
    ln -sf "$(npm prefix -g)/bin/claude" /usr/local/bin/claude
fi
claude --version
INNER

echo "== conda environment $CONDA_ENV =="
if [ -x "$DST/bin/python" ]; then
    echo "environment already exists: $DST"
else
    [ -n "$BASE_ENV" ] || { echo "environment $DST is absent; pass --base-env to clone one" >&2; exit 2; }
    docker exec -i -u root "$CONTAINER" \
        -e "CONDA_SH=${BSKA_NPU_CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}" \
        -e "ENVS_PATH=$ENVS_PATH" -e "CONDA_ENV=$CONDA_ENV" -e "BASE_ENV=$BASE_ENV" \
        bash -s <<'INNER'
set -e
. "$CONDA_SH"
export CONDA_ENVS_PATH="$ENVS_PATH"
conda create -y -p "$ENVS_PATH/$CONDA_ENV" --clone "$ENVS_PATH/$BASE_ENV" --override-channels -c conda-forge
INNER
fi

echo "== python packages + helix =="
docker exec -i -u root "${PROXY_ENV[@]}" -e "PIP_INDEX=$PIP_INDEX" -e "DST=$DST" -e "HELIX_REPO=$HELIX_REPO" \
    "$CONTAINER" bash -s <<'INNER'
set -e
"$DST/bin/python" -m pip install -q pybind11 -i "$PIP_INDEX"
if [ -n "$HELIX_REPO" ]; then
    "$DST/bin/python" -m pip install -q --no-build-isolation --no-deps -e "$HELIX_REPO"
fi
"$DST/bin/python" -c "import pybind11; print('pybind11', pybind11.__version__)"
if [ -n "$HELIX_REPO" ]; then "$DST/bin/helix" --help >/dev/null && echo "helix installed"; fi
INNER

echo "provisioning complete for $CONTAINER"
