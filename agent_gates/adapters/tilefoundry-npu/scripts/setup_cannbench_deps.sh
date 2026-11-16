#!/usr/bin/env bash
# One-time pure-Python dependency installer for CANNBench inside the TileFoundry Conda env.
# Installs ONLY non-Torch dependencies so pinned torch/torch_npu versions are never altered.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${TILEFOUNDRY_NPU_ENV_FILE:-$ADAPTER_DIR/env.sh}"

if [ ! -f "$ENV_FILE" ]; then
  echo "setup_cannbench_deps: machine file not found: $ENV_FILE" >&2
  exit 2
fi

PYTHON_BIN="$(python3 - "$ENV_FILE" <<'PY'
# Strictly parse NAME=VALUE lines as data (never execute env.sh).
import pathlib, re, sys

env_path = pathlib.Path(sys.argv[1])
allowed_re = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
vals = {}
for raw in env_path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    m = allowed_re.match(line)
    if m:
        v = m.group(2).strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        vals[m.group(1)] = v

py = vals.get('TILEFOUNDRY_NPU_PYTHON', '').strip()
if not py:
    cenv = vals.get('TILEFOUNDRY_NPU_CONDA_ENV', '').strip()
    cpath = vals.get('TILEFOUNDRY_NPU_CONDA_ENVS_PATH', '').strip()
    csh = vals.get('TILEFOUNDRY_NPU_CONDA_SH', '').strip()
    if cpath and cenv:
        py = str(pathlib.Path(cpath) / cenv / 'bin' / 'python3')
    elif csh and cenv:
        py = str(pathlib.Path(csh).parent.parent.parent / 'envs' / cenv / 'bin' / 'python3')
print(py)
PY
)"

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "setup_cannbench_deps: resolved python is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi

PRE_VERSIONS="$("$PYTHON_BIN" -c 'import importlib.metadata as md; print(md.version("torch"), md.version("torch-npu"))')"
echo "Pre-install versions: $PRE_VERSIONS ($PYTHON_BIN)"

"$PYTHON_BIN" -m pip install "PyYAML>=6.0" "expecttest" "ruamel.yaml>=0.18" "build" "safetensors>=0.6.2,<1"

POST_VERSIONS="$("$PYTHON_BIN" -c 'import importlib.metadata as md, yaml, ruamel.yaml, expecttest, build, safetensors; print(md.version("torch"), md.version("torch-npu"))')"
if [ "$PRE_VERSIONS" != "$POST_VERSIONS" ]; then
  echo "setup_cannbench_deps: ERROR torch/torch_npu version changed ($PRE_VERSIONS -> $POST_VERSIONS)" >&2
  exit 1
fi
echo "OK: pure-Python CANNBench deps ready; preserved versions: $POST_VERSIONS"
