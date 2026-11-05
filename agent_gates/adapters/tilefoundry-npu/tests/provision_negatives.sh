#!/usr/bin/env bash
# Negatives for provision.py. Each case is a way provisioning could wrongly pass,
# or wrongly REBUILD a shared container.
#
# NO CASE NAMES SOMEONE ELSE'S CONTAINER. Drift is exercised against a RECORDED
# docker inspect document (tests/fixtures/docker-inspect-foreign.json), because a
# peer's container is not ours to inspect, start, stop or rely on: it can vanish,
# change, or belong to a live job. A recording is data, so the case is
# deterministic and cannot disturb anyone. The one real container this script
# creates is our OWN, and it is removed in a finally block before the Id-set
# assertion.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROVISION="$HERE/../scripts/provision.py"
FIXTURE="$HERE/fixtures/docker-inspect-foreign.json"
TMP="$(mktemp -d)"
PY="${PY:-python3}"
pass=0; fail=0
OWNED_CONTAINER="tf-provision-negative-$$"

cleanup() {
  # finally: our own fixture is removed on EVERY exit path. Without this, a failed
  # assertion above could leave a container behind and the Id-set check below would
  # be comparing a world that already contains our leftovers.
  docker rm -f "$OWNED_CONTAINER" >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT

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

echo "== drift: a RECORDED container that is not this deployment =="
"$PY" "$PROVISION" --check --inspect-json "$FIXTURE" > "$TMP/drift.json" 2>&1; rc=$?
[ "$rc" = 3 ] && ok "recorded foreign deployment reports DEPLOYMENT_DRIFT (exit 3)" || bad "exit $rc, wanted 3"
grep -q "DEPLOYMENT_DRIFT" "$TMP/drift.json" && ok "verdict is named DEPLOYMENT_DRIFT" || bad "verdict not named"
grep -q "IMAGE_DRIFT" "$TMP/drift.json" && ok "the image difference is named" || bad "image difference not named"
grep -q "DEVICES_DRIFT" "$TMP/drift.json" && ok "the device difference is named" || bad "device difference not named"
grep -q "MOUNTS_DRIFT" "$TMP/drift.json" && ok "the bind difference is named" || bad "bind difference not named"
grep -q "recorded inspect (no live docker query)" "$TMP/drift.json" \
  && ok "the report says the inspect was a recording" \
  || bad "the report does not distinguish a recording from the live daemon"

echo "== a recording is evidence, never a deployment to mutate =="
"$PY" "$PROVISION" --apply --inspect-json "$FIXTURE" > "$TMP/apply_fixture.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "--apply on a recording exits 2" || bad "--apply on a recording exit $rc, wanted 2"
grep -q "INSPECT_JSON_READ_ONLY" "$TMP/apply_fixture.json" && ok "the refusal is named INSPECT_JSON_READ_ONLY" || bad "refusal not named"

"$PY" "$PROVISION" --check --inspect-json "$TMP/no-such-recording.json" > "$TMP/unreadable.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "an unreadable recording exits 2" || bad "unreadable recording exit $rc, wanted 2"
grep -q "INSPECT_UNREADABLE" "$TMP/unreadable.json" && ok "the refusal is named INSPECT_UNREADABLE" || bad "unreadable refusal not named"

printf '[1, 2]\n' > "$TMP/malformed.json"
"$PY" "$PROVISION" --check --inspect-json "$TMP/malformed.json" > "$TMP/malformed_out.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "a malformed recording exits 2" || bad "malformed recording exit $rc, wanted 2"
grep -q "INSPECT_MALFORMED" "$TMP/malformed_out.json" && ok "the refusal is named INSPECT_MALFORMED" || bad "malformed refusal not named"

echo "== a container WE own: created, inspected, then REMOVED =="
ids_before="$(docker ps -a --format '{{.ID}}' | sort)"
"$PY" "$PROVISION" --apply --container "$OWNED_CONTAINER" > "$TMP/create.json" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "creating our own absent container exits 0" || bad "create exit $rc, wanted 0"
grep -q '"CREATED"' "$TMP/create.json" && ok "verdict names CREATED" || bad "create verdict not named"
if docker inspect "$OWNED_CONTAINER" >/dev/null 2>&1; then
  ok "the container exists after --apply"
else
  bad "the container was not created"
fi
running="$(docker inspect -f '{{.State.Running}}' "$OWNED_CONTAINER" 2>/dev/null)"
[ "$running" = "false" ] && ok "creating does not START it (provisioning is not a device job)" \
  || bad "the created container is running=$running, wanted false"

"$PY" "$PROVISION" --check --container "$OWNED_CONTAINER" > "$TMP/owned.json" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "our own container verifies as PROVISIONED_NOOP" || bad "owned check exit $rc, wanted 0"
grep -q "PROVISIONED_NOOP" "$TMP/owned.json" && ok "the owned container's verdict is PROVISIONED_NOOP" || bad "owned verdict not named"

docker rm -f "$OWNED_CONTAINER" >/dev/null 2>&1
if docker inspect "$OWNED_CONTAINER" >/dev/null 2>&1; then
  bad "our own container still exists after cleanup"
else
  ok "our own container no longer exists after cleanup"
fi
ids_after="$(docker ps -a --format '{{.ID}}' | sort)"
if [ "$ids_before" = "$ids_after" ]; then
  ok "docker ps -a Id set is identical before and after"
else
  bad "docker ps -a Id set changed"
  diff <(printf '%s\n' "$ids_before") <(printf '%s\n' "$ids_after") | sed 's/^/        /'
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
