#!/usr/bin/env bash
# Negatives for provision.py. Each case is a way provisioning could wrongly pass,
# or wrongly REBUILD a shared container.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROVISION="$HERE/../scripts/provision.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PY="${PY:-python3}"
pass=0; fail=0
ok()  { printf "  PASS  %s\n" "$1"; pass=$((pass + 1)); }
bad() { printf "  FAIL  %s\n" "$1"; fail=$((fail + 1)); }

echo "== idempotence: two checks produce the SAME bytes =="
"$PY" "$PROVISION" --check > "$TMP/a.json" 2>&1; rc1=$?
"$PY" "$PROVISION" --check > "$TMP/b.json" 2>&1; rc2=$?
if [ "$rc1" = 0 ] && [ "$rc2" = 0 ] && cmp -s "$TMP/a.json" "$TMP/b.json"; then
  ok "second run is a noop with identical output"
else
  bad "second run differs (rc1=$rc1 rc2=$rc2)"
fi

echo "== absent container =="
"$PY" "$PROVISION" --check --container tf-no-such-container-xyz > "$TMP/absent.json" 2>&1
rc=$?
[ "$rc" = 4 ] && ok "absent container exits 4 (CONTAINER_ABSENT)" || bad "absent container exit $rc, wanted 4"
grep -q "CONTAINER_ABSENT" "$TMP/absent.json" && ok "verdict names CONTAINER_ABSENT" || bad "verdict not named"

echo "== drift: a real container that is NOT this deployment =="
before="$(docker inspect -f "{{.Id}} {{.State.StartedAt}}" netdata 2>/dev/null)"
if [ -z "$before" ]; then
  echo "  SKIP  no netdata container to inspect (drift negative not exercised)"
else
  "$PY" "$PROVISION" --check --container netdata > "$TMP/drift.json" 2>&1; rc=$?
  [ "$rc" = 3 ] && ok "foreign container reports DEPLOYMENT_DRIFT (exit 3)" || bad "exit $rc, wanted 3"
  grep -q "DEPLOYMENT_DRIFT" "$TMP/drift.json" && ok "verdict is named DEPLOYMENT_DRIFT" || bad "verdict not named"
  grep -q "IMAGE_DRIFT" "$TMP/drift.json" && ok "the image difference is named" || bad "image difference not named"
  after="$(docker inspect -f "{{.Id}} {{.State.StartedAt}}" netdata 2>/dev/null)"
  [ "$before" = "$after" ] && ok "the inspected container was NOT touched" || bad "the container changed"

  echo "== --apply on an existing (drifting) container must refuse to rebuild =="
  before_apply="$(docker inspect -f "{{.Id}}" netdata)"
  "$PY" "$PROVISION" --apply --container netdata > "$TMP/apply.json" 2>&1; rc=$?
  [ "$rc" = 3 ] && ok "--apply refuses with exit 3" || bad "--apply exit $rc, wanted 3"
  grep -q "will NOT remove or recreate" "$TMP/apply.json" && ok "refusal states no rebuild" || bad "refusal does not say it will not rebuild"
  [ "$before_apply" = "$(docker inspect -f "{{.Id}}" netdata)" ] && ok "container id unchanged (nothing recreated)" || bad "container was replaced"
fi

echo "== image tag resolving to a different digest =="
"$PY" - "$HERE/../deployment_profile.json" "$TMP/bad_profile.json" <<'PY'
import json, sys
profile = json.load(open(sys.argv[1]))
profile["required_image_digest"] = "sha256:" + "0" * 64
json.dump(profile, open(sys.argv[2], "w"))
PY
"$PY" "$PROVISION" --check --profile "$TMP/bad_profile.json" > "$TMP/tagmoved.json" 2>&1; rc=$?
[ "$rc" = 3 ] && ok "a moved/incorrect digest is refused (exit 3)" || bad "exit $rc, wanted 3"
grep -q "IMAGE_TAG_MOVED" "$TMP/tagmoved.json" && ok "verdict names IMAGE_TAG_MOVED" || bad "verdict not named"

echo "== --apply --dry-run on an absent container creates nothing =="
"$PY" "$PROVISION" --apply --dry-run --container tf-dry-run-probe > "$TMP/dry.json" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "dry run exits 0" || bad "dry run exit $rc"
grep -q "create_argv" "$TMP/dry.json" && ok "the create argv is reported" || bad "no create argv"
if docker inspect tf-dry-run-probe >/dev/null 2>&1; then bad "dry run created a container"; else ok "nothing was created"; fi

echo
echo "passed: $pass, failed: $fail"
[ "$fail" -eq 0 ] && { echo "RESULT: PASS"; exit 0; }
echo "RESULT: FAIL"; exit 1
