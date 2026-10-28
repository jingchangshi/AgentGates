#!/usr/bin/env bash
#
# First-time AgentGates configuration and first run.
#
# This script performs the README's initial configuration flow as one
# parameterized command: it creates the virtualenv, installs the CLI, then
# runs `list`, `doctor`, one full gate run, and the receipt's `report` and
# `verify`. Every machine value — compiler checkout, machine env file,
# adapter, interpreter, venv location, evidence store, NPU device — is
# supplied as a parameter; nothing is guessed and nothing is sourced.
#
# Exit codes follow the CLI contract:
#   0  every step reported PASS
#   1  a recorded non-PASS result (doctor not ready, or report/verify not PASS)
#   2  an invocation or environment error (bad parameters, missing tools)
# The script stops at the first failing step.
#
# How the three bundled adapters are configured on this machine
# --------------------------------------------------------------
# Every adapter reads machine configuration as DATA (NAME=value lines only,
# no command substitution, no sourcing) from its own env.sh:
#
#   ascendnpu-ir       agent_gates/adapters/ascendnpu-ir/env.sh
#   tilelang-hivm      agent_gates/adapters/tilelang-hivm/env.sh
#   tilelang-hivm-npu  shares the tilelang-hivm file
#
# Both files are git-ignored and live in this checkout; the editable install
# below is what puts them on the adapters' discovery path. A launching shell's
# exports are never a configuration source — a value exported in conflict with
# the file is refused, not silently preferred. --env-file only selects a
# different file; it supplies no values.
#
# ascendnpu-ir (gates: ascend-build, ascend-regression) enters the same
# long-running container as the TileLang adapters, because that container
# bind-mounts /home and therefore sees both the source tree and ccec:
#   ASCENDNPU_CONTAINER_NAME=s00653124-tilelang-hivm
#   ASCENDNPU_CONTAINER_USER=s00653124
#   ASCENDNPU_CONTAINER_HOME=/home/s00653124
#   ASCENDNPU_BUILD_SUBDIR=build        ASCENDNPU_NINJA_JOBS=64
#   ASCENDNPU_CCEC_BIN=/home/s00653124/workspace/ccec-toolkits/default/ccec_compiler/bin
#   ASCENDNPU_CANN_SETUP=/data/pri/cann-9.1.0/set_env.sh
#   ASCENDNPU_CONDA_SH=  ASCENDNPU_CONDA_ENV=   (container system Python 3.12)
#
# tilelang-hivm / tilelang-hivm-npu (gates: tilelang-hivm-build,
# tilelang-hivm-regression, tilelang-hivm-npu-sanity) use the same container
# and user, with CANN plus a conda env as the toolchain:
#   TILELANG_HIVM_CONTAINER_NAME=s00653124-tilelang-hivm
#   TILELANG_HIVM_CONTAINER_USER=s00653124
#   TILELANG_HIVM_CONTAINER_HOME=/home/s00653124
#   TILELANG_HIVM_CANN_SETUP=/data/pri/cann-9.1.0/set_env.sh
#   TILELANG_HIVM_CONDA_SH=/data/miniconda3/etc/profile.d/conda.sh
#   TILELANG_HIVM_CONDA_ENV=s00653124_tilelang
#   TILELANG_HIVM_CONDA_ENVS_PATH=/home/s00653124/.conda/envs
#   TILELANG_HIVM_HIVM_COMPILER=/home/s00653124/workspace/AscendNPU-IR-1/build/bin/bishengir-compile
#   TILELANG_HIVM_HIVM_TARGET=Ascend950PR_9579
# TILELANG_HIVM_DEVICE_ID is deliberately absent from the file: a device
# authorizes one run, so the NPU adapter needs --device-id; a card that
# merely looks idle never authorizes its use.
#
# Per-adapter invocations (python3.11 is not on PATH here; it lives at
# /data/miniconda3/bin/python3.11, or rely on the python3 fallback):
#   ascendnpu-ir:      scripts/bootstrap.sh --adapter ascendnpu-ir --source /abs/AscendNPU-IR
#   tilelang-hivm:     scripts/bootstrap.sh --adapter tilelang-hivm --source /abs/tilelang
#   tilelang-hivm-npu: scripts/bootstrap.sh --adapter tilelang-hivm-npu --source /abs/tilelang --device-id 0

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/bootstrap.sh --source ROOT [options]

Performs the first-time AgentGates setup: virtualenv, editable install,
then list / doctor / run --all / report / verify with the machine values
given as parameters.

Required:
  --source ROOT          compiler checkout root (absolute path, repository root)

Machine configuration:
  --env-file PATH        machine env file for the adapter (absolute path);
                         omit to use the adapter's bundled env.sh discovery
  --adapter ID           adapter to configure (default: tilelang-hivm)
  --device-id ID         explicitly authorize this NPU device for the run;
                         required for tilelang-hivm-npu

Installation:
  --python CMD           interpreter used to create the virtualenv
                         (default: python3.11)
  --venv-dir PATH        virtualenv directory, relative paths resolve against
                         the repository root (default: .venv)
  --recreate-venv        delete and recreate the virtualenv first

Run selection and evidence:
  --gate ID              run this gate instead of --all (repeatable)
  --evidence-root PATH   evidence store (default: ~/.local/state/agent-gates)

  -h, --help             show this help
EOF
}

ADAPTER="tilelang-hivm"
PYTHON="python3.11"
PYTHON_EXPLICIT=0
VENV_DIR=".venv"
SOURCE=""
ENV_FILE=""
EVIDENCE_ROOT=""
DEVICE_ID=""
GATES=()
RECREATE_VENV=0

while (( $# )); do
    case "$1" in
        --adapter)       ADAPTER="${2:?--adapter needs a value}"; shift 2 ;;
        --source)        SOURCE="${2:?--source needs a value}"; shift 2 ;;
        --env-file)      ENV_FILE="${2:?--env-file needs a value}"; shift 2 ;;
        --python)        PYTHON="${2:?--python needs a value}"; PYTHON_EXPLICIT=1; shift 2 ;;
        --venv-dir)      VENV_DIR="${2:?--venv-dir needs a value}"; shift 2 ;;
        --evidence-root) EVIDENCE_ROOT="${2:?--evidence-root needs a value}"; shift 2 ;;
        --device-id)     DEVICE_ID="${2:?--device-id needs a value}"; shift 2 ;;
        --gate)          GATES+=("${2:?--gate needs a value}"); shift 2 ;;
        --recreate-venv) RECREATE_VENV=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "bootstrap: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

fail() {
    echo "bootstrap: $*" >&2
    exit 2
}

step() {
    printf '\n==> %s\n' "$*"
}

# --- validate parameters ------------------------------------------------------

[ -n "$SOURCE" ] || fail "--source is required (the compiler checkout root)"
case "$SOURCE" in
    /*) ;;
    *) fail "--source must be an absolute path: $SOURCE" ;;
esac
[ -d "$SOURCE" ] || fail "--source is not a directory: $SOURCE"

if [ -n "$ENV_FILE" ]; then
    case "$ENV_FILE" in
        /*) ;;
        *) fail "--env-file must be an absolute path: $ENV_FILE" ;;
    esac
    [ -f "$ENV_FILE" ] || fail "--env-file is not a file: $ENV_FILE"
fi

# The interpreter must exist and speak Python >= 3.11; the interpreter itself
# decides, so no version-string parsing is trusted. When the default
# python3.11 is absent and python3 is new enough, fall back to it once with a
# visible notice; an explicitly requested interpreter is never substituted.
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    if [ "$PYTHON_EXPLICIT" -eq 0 ] && command -v python3 >/dev/null 2>&1 \
        && [ "$(python3 -c 'import sys; print(int(sys.version_info >= (3, 11)))')" = "1" ]; then
        echo "bootstrap: python3.11 not found; using python3 ($(python3 -V 2>&1))" >&2
        PYTHON=python3
    else
        fail "interpreter not found: $PYTHON (use --python /path/to/python3.11)"
    fi
fi
if [ "$("$PYTHON" -c 'import sys; print(int(sys.version_info >= (3, 11)))')" != "1" ]; then
    fail "$PYTHON is $("$PYTHON" -V 2>&1); AgentGates requires Python >= 3.11"
fi

case "$VENV_DIR" in
    /*) VENV_PATH="$VENV_DIR" ;;
    *) VENV_PATH="$REPO_ROOT/$VENV_DIR" ;;
esac

# --- 1. virtualenv and editable install ---------------------------------------

if [ "$RECREATE_VENV" -eq 1 ] && [ -e "$VENV_PATH" ]; then
    step "Recreating virtualenv at $VENV_PATH"
    rm -rf "$VENV_PATH" || fail "could not remove $VENV_PATH"
fi

if [ ! -x "$VENV_PATH/bin/python" ]; then
    step "Creating virtualenv at $VENV_PATH with $PYTHON ($("$PYTHON" -V 2>&1))"
    "$PYTHON" -m venv "$VENV_PATH" || fail "venv creation failed"
fi

VENV_PY="$VENV_PATH/bin/python"
AG="$VENV_PATH/bin/agent-gates"

step "Installing AgentGates into $VENV_PATH"
(
    cd "$REPO_ROOT"
    "$VENV_PY" -m pip install -e .
) || fail "pip install -e . failed"
[ -x "$AG" ] || fail "$AG missing after install; retry with --recreate-venv"

# Common options shared by doctor and run. Every value was validated above,
# so the CLI receives absolute paths and the same store on every command.
COMMON=(--adapter "$ADAPTER" --source "$SOURCE")
if [ -n "$ENV_FILE" ]; then
    COMMON+=(--env-file "$ENV_FILE")
fi
if [ -n "$EVIDENCE_ROOT" ]; then
    COMMON+=(--evidence-root "$EVIDENCE_ROOT")
fi
if [ -n "$DEVICE_ID" ]; then
    COMMON+=(--device-id "$DEVICE_ID")
fi

# --- 2. list ------------------------------------------------------------------

step "Listing bundled adapters and gates"
"$AG" list

# --- 3. doctor ----------------------------------------------------------------

step "Checking the machine configuration for adapter $ADAPTER"
doctor_status=0
"$AG" doctor "${COMMON[@]}" || doctor_status=$?
if [ "$doctor_status" -ne 0 ]; then
    echo "bootstrap: doctor did not report ready (exit $doctor_status);" \
         "fix the machine configuration before running gates" >&2
    exit "$doctor_status"
fi

# --- 4. run -------------------------------------------------------------------

if [ "${#GATES[@]}" -gt 0 ]; then
    GATE_ARGS=()
    for gate in "${GATES[@]}"; do
        GATE_ARGS+=(--gate "$gate")
    done
else
    GATE_ARGS=(--all)
fi

RUN_JSON="$(mktemp)" || fail "could not create a temporary file"
trap 'rm -f "$RUN_JSON"' EXIT

step "Running gates for adapter $ADAPTER"
run_status=0
"$AG" run "${COMMON[@]}" "${GATE_ARGS[@]}" --json | tee "$RUN_JSON" || run_status=$?

RUN_ID="$("$VENV_PY" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["runId"])' "$RUN_JSON" 2>/dev/null)" || RUN_ID=""
if [ -z "$RUN_ID" ]; then
    echo "bootstrap: could not read runId from the run output" >&2
    exit 2
fi
if [ "$run_status" -ne 0 ]; then
    echo "bootstrap: run $RUN_ID did not reach PASS (exit $run_status);" \
         "inspect the receipt before reporting or verifying" >&2
    exit "$run_status"
fi

# --- 5. report and verify -----------------------------------------------------

step "Reporting run $RUN_ID"
report_status=0
"$AG" report --run-id "$RUN_ID" --json || report_status=$?

step "Verifying run $RUN_ID"
verify_status=0
"$AG" verify --run-id "$RUN_ID" --json || verify_status=$?

final_status=0
if [ "$report_status" -ne 0 ]; then
    final_status=$report_status
elif [ "$verify_status" -ne 0 ]; then
    final_status=$verify_status
fi

if [ "$final_status" -eq 0 ]; then
    printf '\nbootstrap: complete — adapter %s, run %s, receipt PASS\n' "$ADAPTER" "$RUN_ID"
else
    printf '\nbootstrap: finished with a recorded non-PASS result — adapter %s, run %s (exit %d)\n' \
        "$ADAPTER" "$RUN_ID" "$final_status" >&2
fi
exit "$final_status"
