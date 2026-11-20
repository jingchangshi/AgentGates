#!/usr/bin/env bash
# Host control process entry point.
#
# -S keeps site initialization (sitecustomize and .pth files) out of the
# adapter, and -E ignores inherited PYTHON* variables, so a project startup
# hook cannot load before the adapter builds and scrubs the target
# environment. The workload interpreter is activated separately by the
# adapter.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 -S -E -B "$here/tilefoundry_host_gate.py" "$@"
