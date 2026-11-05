#!/usr/bin/env bash
# Negatives for the tilefoundry-npu adapter.
#
# Each case is a way the Gate could wrongly pass. A test that only exercises the
# happy path proves the code runs, not that it refuses.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/../scripts/tilefoundry_gate.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# LOCK HYGIENE: the --device negatives below invoke the REAL Gate, which enters the
# device lease BEFORE it validates its arguments. Without this override every one of
# them would acquire and release the canonical team lock
# /home/.tf_device0.lock -- the file that coordinates the whole team. Running the
# tests must never take it. TF_DEVICE_LOCK_PATH is the Gate's own explicit override
# for exactly this; no negative here asserts anything about the canonical path.
export TF_DEVICE_LOCK_PATH="$TMP/.tf-device-negatives.lock"
pass=0; fail=0

ok()   { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }

# expect CODE WANT LABEL -- command...
expect() {
    local want="$1" label="$2"; shift 2
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ "$rc" = "$want" ]; then
        ok "$label (exit $rc)"
    else
        bad "$label (exit $rc, wanted $want)"
        printf '%s\n' "$out" | sed 's/^/        /' | head -6
    fi
}

echo "== configuration negatives =="

# A file that does not exist must be refused, not defaulted.
expect 2 "missing env file refused" \
    python3 -B "$GATE" --check-config --env-file "$TMP/absent.sh"

# A key the Gate does not know must fail: a silently ignored typo is a value
# the operator believes in and the Gate does not have.
cat > "$TMP/unknown.sh" <<'EOS'
TF_REPO=/home/s00653124/workspace/TileFoundry
IMAGE=x
CONTAINER_NAME=y
CANN_ROOT=/c
IN_CONTAINER_PYTHON=/p
CONTAINER_USER=u
NOT_A_KEY=1
EOS
expect 2 "unknown key refused" \
    python3 -B "$GATE" --check-config --env-file "$TMP/unknown.sh"

# A duplicate key is ambiguous and must fail.
cat > "$TMP/dup.sh" <<'EOS'
TF_REPO=/home/s00653124/workspace/TileFoundry
TF_REPO=/somewhere/else
EOS
expect 2 "duplicate key refused" \
    python3 -B "$GATE" --check-config --env-file "$TMP/dup.sh"

# A required value left empty must fail rather than be filled in.
cat > "$TMP/incomplete.sh" <<'EOS'
TF_REPO=/home/s00653124/workspace/TileFoundry
IMAGE=
CONTAINER_NAME=y
CANN_ROOT=/c
IN_CONTAINER_PYTHON=/p
CONTAINER_USER=u
EOS
expect 2 "missing required value refused" \
    python3 -B "$GATE" --check-config --env-file "$TMP/incomplete.sh"

# A repo without the docker recipe must fail: this adapter reuses that recipe,
# so a repo that lacks it is not a TileFoundry checkout it can drive.
cat > "$TMP/norecipe.sh" <<'EOS'
TF_REPO=/tmp
IMAGE=x
CONTAINER_NAME=y
CANN_ROOT=/c
IN_CONTAINER_PYTHON=/p
CONTAINER_USER=u
EOS
expect 2 "repo without the docker recipe refused" \
    python3 -B "$GATE" --check-config --env-file "$TMP/norecipe.sh"

echo "== host guard negatives =="

# The host must be refused. This test necessarily runs on the host, which is
# exactly the condition being tested.
cat > "$TMP/good.sh" <<'EOS'
TF_REPO=/home/s00653124/workspace/TileFoundry
IMAGE=tilefoundry-ascend950pr-dev:24.04
CONTAINER_NAME=tf-ascend950pr
CANN_ROOT=/data/pri/cann-9.2.0
IN_CONTAINER_PYTHON=/home/s00653124/.conda/envs/tilefoundry-npu/bin/python
CONTAINER_USER=s00653124
EOS
expect 2 "running the CPU Gate on the host is refused" \
    python3 -B "$GATE" --cpu --env-file "$TMP/good.sh"

out="$(python3 -B "$GATE" --cpu --env-file "$TMP/good.sh" 2>&1)"
if printf '%s' "$out" | grep -q "refusing to run the CPU Gate on the HOST"; then
    ok "the host refusal names the operation"
else
    bad "the host refusal does not say what it refused"
fi
if printf '%s' "$out" | grep -q "docker exec"; then
    ok "the host refusal says how to run it correctly"
else
    bad "the host refusal does not offer the correct invocation"
fi

echo "== device negatives =="

expect 2 "device Gate without a device id is refused" \
    python3 -B "$GATE" --device --env-file "$TMP/good.sh"

expect 2 "non-numeric device list is refused" \
    python3 -B "$GATE" --device --device-id banana --env-file "$TMP/good.sh"

expect 2 "duplicate device list is refused" \
    python3 -B "$GATE" --device --device-id 0,0 --env-file "$TMP/good.sh"

expect 2 "empty device list is refused" \
    python3 -B "$GATE" --device --device-id "" --env-file "$TMP/good.sh"

echo "== no Gate selected =="
expect 2 "no subcommand refuses rather than guessing" \
    python3 -B "$GATE" --env-file "$TMP/good.sh"

echo "== retired paths must not reappear =="

# /opt/tf-venv is deprecated. A stale reference in a fixture or a doc is how a
# retired interpreter quietly comes back, so this is checked mechanically rather
# than trusted to review. Deprecation NOTICES are allowed; USE is not.
stale=0
for f in "$HERE/../scripts"/* "$HERE/../env.example.sh"; do
    [ -f "$f" ] || continue
    hits="$(grep -n 'opt/tf-venv' "$f" 2>/dev/null | grep -vi 'DEPRECATED\|deprecated\|no longer' | head -3)"
    if [ -n "$hits" ]; then
        echo "  FAIL  $f still uses the retired /opt interpreter"
        printf '%s\n' "$hits" | sed 's/^/        /'
        stale=$((stale + 1))
    fi
done
if [ "$stale" -eq 0 ]; then
    ok "no live /opt/tf-venv reference in scripts or env"
else
    fail=$((fail + stale))
fi

echo "== shared-card policy: record only, no wait or refusal =="

expect 0 "busy and idle run the same workload with the same verdict" \
    python3 "$HERE/device_policy.py"
echo "== a non-interpreter must be refused, not passed =="

# The Architect's FALSE PASS: IN_CONTAINER_PYTHON=/bin/true gave exit 0 with every
# field true and nothing ran. This proves the identity check refuses /bin/true.
expect 0 "interpreter identity refuses /bin/true" \
    python3 "$HERE/identity_check.py"

echo "== exit 0 with no structured report is a failure =="

expect 0 "tool reports must be structured PASS, not bare exit 0" \
    python3 "$HERE/tool_report_check.py"

echo "== manifest and script agree on every check =="

# Static, so it needs no container. Drift between the declared observationChecks and
# what the script reports is invisible in a normal run: both sides look like success.
expect 0 "manifest observationChecks match the script" \
    python3 "$HERE/manifest_parity.py"

echo
echo "passed: $pass, failed: $fail"
[ "$fail" -eq 0 ] && { echo "RESULT: PASS"; exit 0; }
echo "RESULT: FAIL"; exit 1