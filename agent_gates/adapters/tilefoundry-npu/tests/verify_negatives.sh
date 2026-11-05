#!/usr/bin/env bash
# The seven ways verify_environment.py could report a green it did not earn.
# Each case injects ONE fault and asserts the NAMED refusal, not merely a non-zero exit.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
VERIFY="$HERE/../scripts/verify_environment.py"
PY="$PY"
OUT="$(mktemp -d)"
TMPD="$OUT"
# NEVER take the canonical /home/.tf_device0.lock. These negatives are not team
# measurement jobs; they point the lock at a private path so running them cannot
# block, or be blocked by, a real one. The override is recorded by the script.
export TF_DEVICE_LOCK_PATH="$OUT/.tf-device-negatives.lock"
trap 'rm -rf "$OUT"' EXIT
pass=0; fail=0
if [ -z "$PY" ]; then echo "set PY to the deployment interpreter" >&2; exit 2; fi
ok()  { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad() { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }

export ASCEND_RT_VISIBLE_DEVICES=1
export TFNU_WHEELS=/home/s00653124/workspace/envs/wheels-main
export TFNU_EXPECTED_TILELANG_VERSION=0.1.15+cpu.git0e687c17

has_code() {
  if grep -q "$2" "$1"/*/evidence.json 2>/dev/null; then ok "$3"; else bad "$3 (code $2 missing)"; fi
}

echo "== 1. CANN not sourced =="
( unset ASCEND_HOME_PATH LD_LIBRARY_PATH; "$PY" "$VERIFY" --out "$TMPD/cann" --side negative > "$TMPD/cann.log" 2>&1; echo "$?" > "$TMPD/cann.rc" )
rc="$(cat "$TMPD/cann.rc")"
[ "$rc" = 75 ] && ok "missing CANN exits 75" || bad "missing CANN exit $rc, wanted 75"
has_code "$TMPD/cann" CANN_NOT_SOURCED "code is CANN_NOT_SOURCED"

echo "== 2. tilelang source tree on the import path =="
PYTHONPATH=/home/s00653124/workspace/tilelang "$PY" "$VERIFY" --out "$TMPD/tree" --side negative > "$TMPD/tree.log" 2>&1
rc="$?"
[ "$rc" = 1 ] && ok "source tree exits 1" || bad "source tree exit $rc, wanted 1"
has_code "$TMPD/tree" TILELANG_NOT_FROM_PREFIX "code is TILELANG_NOT_FROM_PREFIX"

echo "== 3. cache dir is not empty =="
mkdir -p "$TMPD/cache"
echo stale > "$TMPD/cache/leftover.bin"
"$PY" "$VERIFY" --out "$TMPD/cacheout" --side negative --cache-dir "$TMPD/cache" --skip-device > "$TMPD/cache.log" 2>&1
rc="$?"
[ "$rc" = 1 ] && ok "non-empty cache exits 1" || bad "non-empty cache exit $rc, wanted 1"
has_code "$TMPD/cacheout" CACHE_NOT_EMPTY "code is CACHE_NOT_EMPTY"

echo "== 4. wheel version mismatch =="
TFNU_EXPECTED_TILELANG_VERSION=0.0.0+nope "$PY" "$VERIFY" --out "$TMPD/ver" --side negative --skip-device > "$TMPD/ver.log" 2>&1
rc="$?"
[ "$rc" = 1 ] && ok "version mismatch exits 1" || bad "version mismatch exit $rc, wanted 1"
has_code "$TMPD/ver" TILELANG_WHEEL_VERSION_MISMATCH "code is TILELANG_WHEEL_VERSION_MISMATCH"

echo "== 5. logical index outside the visible list =="
"$PY" "$VERIFY" --out "$TMPD/idx" --side negative --device-index 1 > "$TMPD/idx.log" 2>&1
rc="$?"
[ "$rc" = 75 ] && ok "out-of-range index exits 75" || bad "out-of-range index exit $rc, wanted 75"
has_code "$TMPD/idx" DEVICE_INDEX_NOT_VISIBLE "code is DEVICE_INDEX_NOT_VISIBLE"

echo "== 6. one-bit error injected into the reference =="
"$PY" "$VERIFY" --out "$TMPD/inj" --side negative --inject-error 1 > "$TMPD/inj.log" 2>&1
rc="$?"
[ "$rc" = 1 ] && ok "injected error exits 1" || bad "injected error exit $rc, wanted 1"
has_code "$TMPD/inj" VADD_MISMATCH "code is VADD_MISMATCH"

echo "== 7. a configure that WRITES to the environment =="
FAKE="$TMPD/fake_configure.py"
cat > "$FAKE" <<'EOPY'
import json
import pathlib
import sys

if "--findings" not in sys.argv:
    raise SystemExit(2)
site = pathlib.Path(sys.executable).resolve().parents[1] / "lib" / "python3.12" / "site-packages"
d = site / "verify_negative_probe-0.0.1.dist-info"
if not d.exists():
    d.mkdir()
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verify-negative-probe\nVersion: 0.0.1\n")
print(json.dumps({"actions": ["noop"]}))
EOPY
"$PY" "$VERIFY" --out "$TMPD/mut" --side negative --skip-device --configure-cmd "$PY $FAKE" > "$TMPD/mut.log" 2>&1
rc="$?"
PURELIB="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
rm -rf "$PURELIB/verify_negative_probe-0.0.1.dist-info"
[ "$rc" = 1 ] && ok "mutating configure exits 1" || bad "mutating configure exit $rc, wanted 1"
has_code "$TMPD/mut" FINDINGS_MUTATED_ENV "code is FINDINGS_MUTATED_ENV"

echo
echo "passed: $pass, failed: $fail"
[ "$fail" -eq 0 ] && { echo "RESULT: PASS"; exit 0; }
echo "RESULT: FAIL"; exit 1
