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
#
# CLEAN-ARCHIVE RULE. This script must pass from `git archive <sha> | tar -x`, where
# the machine-local, git-ignored env.sh does NOT exist. Every case therefore names
# the checked-in fixture tests/fixtures/provision.env.fixture explicitly, and the
# idempotence case runs on a RECORDED inspect document generated from the tracked
# deployment profile -- never on the current machine's live deployment. A suite that
# only passes in a working tree with an env.sh is not reproducible; measured
# failure: all cases died with rc=2 (ENV_UNREADABLE) in a clean archive.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROVISION="$HERE/../scripts/provision.py"
PROFILE="$HERE/../deployment_profile.json"
ENV_FIXTURE="$HERE/fixtures/provision.env.fixture"
DRIFT_FIXTURE="$HERE/fixtures/docker-inspect-foreign.json"
TMP="$(mktemp -d)"
PY="${PY:-python3}"
# Every provision.py call names the fixture, never env.sh.
ENV_ARGS=(--env-file "$ENV_FIXTURE")
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

# A RECORDED inspect document that DOES match the tracked profile, generated from
# the profile itself so it cannot go stale. This replaces the old "check the live
# deployment twice" case, which required a machine-local env.sh and a live
# container -- neither exists in a clean archive.
"$PY" - "$PROFILE" "$TMP/matching.json" <<'PY'
import json, sys
profile = json.load(open(sys.argv[1]))
binds = []
for want in profile["required_binds"]:
    destination = want["destination"]
    source = (destination if profile.get("bind_source_equals_destination", True)
              else want.get("source"))
    binds.append("%s:%s%s" % (source, destination, ":ro" if want.get("read_only") else ""))
inspect = [{
    "Id": "f" * 64,
    "Image": profile["required_image_digest"],
    "Config": {"User": profile.get("required_container_user", "root")},
    "HostConfig": {
        "Binds": binds,
        "Devices": [{"PathOnHost": device} for device in profile["required_devices"]],
    },
    "State": {"Running": False, "StartedAt": "1970-01-01T00:00:00Z"},
}]
json.dump(inspect, open(sys.argv[2], "w"))
PY

echo "== idempotence: two checks of the SAME recorded deployment are byte-identical =="
"$PY" "$PROVISION" --check --inspect-json "$TMP/matching.json" "${ENV_ARGS[@]}" > "$TMP/a.json" 2>&1; rc1=$?
"$PY" "$PROVISION" --check --inspect-json "$TMP/matching.json" "${ENV_ARGS[@]}" > "$TMP/b.json" 2>&1; rc2=$?
if [ "$rc1" = 0 ] && [ "$rc2" = 0 ] && cmp -s "$TMP/a.json" "$TMP/b.json"; then
  ok "second run is a noop with identical output"
else
  bad "second run differs (rc1=$rc1 rc2=$rc2)"
  sed 's/^/        /' "$TMP/a.json" | head -5
fi
grep -q "PROVISIONED_NOOP" "$TMP/a.json" && ok "a matching recorded deployment is PROVISIONED_NOOP" || bad "matching deployment not accepted"

echo "== the env file can come from TFNU_ENV_FILE (no env.sh needed) =="
TFNU_ENV_FILE="$ENV_FIXTURE" "$PY" "$PROVISION" --check --inspect-json "$TMP/matching.json" > "$TMP/envvar.json" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "TFNU_ENV_FILE is honoured" || bad "TFNU_ENV_FILE exit $rc, wanted 0"
grep -q "PROVISIONED_NOOP" "$TMP/envvar.json" && ok "TFNU_ENV_FILE run reaches the verdict" || bad "TFNU_ENV_FILE verdict missing"

echo "== absent container =="
"$PY" "$PROVISION" --check --container tf-no-such-container-xyz "${ENV_ARGS[@]}" > "$TMP/absent.json" 2>&1
rc=$?
[ "$rc" = 4 ] && ok "absent container exits 4 (CONTAINER_ABSENT)" || bad "absent container exit $rc, wanted 4"
grep -q "CONTAINER_ABSENT" "$TMP/absent.json" && ok "verdict names CONTAINER_ABSENT" || bad "verdict not named"

echo "== drift: a RECORDED container that is not this deployment =="
"$PY" "$PROVISION" --check --inspect-json "$DRIFT_FIXTURE" "${ENV_ARGS[@]}" > "$TMP/drift.json" 2>&1; rc=$?
[ "$rc" = 3 ] && ok "recorded foreign deployment reports DEPLOYMENT_DRIFT (exit 3)" || bad "exit $rc, wanted 3"
grep -q "DEPLOYMENT_DRIFT" "$TMP/drift.json" && ok "verdict is named DEPLOYMENT_DRIFT" || bad "verdict not named"
grep -q "IMAGE_DRIFT" "$TMP/drift.json" && ok "the image difference is named" || bad "image difference not named"
grep -q "DEVICES_DRIFT" "$TMP/drift.json" && ok "the device difference is named" || bad "device difference not named"
grep -q "MOUNTS_DRIFT" "$TMP/drift.json" && ok "the bind difference is named" || bad "bind difference not named"
grep -q "recorded inspect (no live docker query)" "$TMP/drift.json" \
  && ok "the report says the inspect was a recording" \
  || bad "the report does not distinguish a recording from the live daemon"

echo "== a recording is evidence, never a deployment to mutate =="
"$PY" "$PROVISION" --apply --inspect-json "$DRIFT_FIXTURE" "${ENV_ARGS[@]}" > "$TMP/apply_fixture.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "--apply on a recording exits 2" || bad "--apply on a recording exit $rc, wanted 2"
grep -q "INSPECT_JSON_READ_ONLY" "$TMP/apply_fixture.json" && ok "the refusal is named INSPECT_JSON_READ_ONLY" || bad "refusal not named"

"$PY" "$PROVISION" --check --inspect-json "$TMP/no-such-recording.json" "${ENV_ARGS[@]}" > "$TMP/unreadable.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "an unreadable recording exits 2" || bad "unreadable recording exit $rc, wanted 2"
grep -q "INSPECT_UNREADABLE" "$TMP/unreadable.json" && ok "the refusal is named INSPECT_UNREADABLE" || bad "unreadable refusal not named"

printf '[1, 2]\n' > "$TMP/malformed.json"
"$PY" "$PROVISION" --check --inspect-json "$TMP/malformed.json" "${ENV_ARGS[@]}" > "$TMP/malformed_out.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "a malformed recording exits 2" || bad "malformed recording exit $rc, wanted 2"
grep -q "INSPECT_MALFORMED" "$TMP/malformed_out.json" && ok "the refusal is named INSPECT_MALFORMED" || bad "malformed refusal not named"

echo "== a container WE own: created, inspected, then REMOVED =="
ids_before="$(docker ps -a --format '{{.ID}}' | sort)"
"$PY" "$PROVISION" --apply --container "$OWNED_CONTAINER" "${ENV_ARGS[@]}" > "$TMP/create.json" 2>&1; rc=$?
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

"$PY" "$PROVISION" --check --container "$OWNED_CONTAINER" "${ENV_ARGS[@]}" > "$TMP/owned.json" 2>&1; rc=$?
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

echo "== an unknown REQUIRED profile key is refused, not ignored =="
"$PY" - "$PROFILE" "$TMP/unknown_key_profile.json" <<'PY'
import json, sys
profile = json.load(open(sys.argv[1]))
profile["required_magic_gpu"] = True
json.dump(profile, open(sys.argv[2], "w"))
PY
"$PY" "$PROVISION" --check --profile "$TMP/unknown_key_profile.json" "${ENV_ARGS[@]}" > "$TMP/unknown.json" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "an unknown required key exits 2" || bad "unknown required key exit $rc, wanted 2"
grep -q "PROFILE_UNKNOWN_KEY" "$TMP/unknown.json" && ok "the refusal is named PROFILE_UNKNOWN_KEY" || bad "unknown-key refusal not named"
grep -q "required_magic_gpu" "$TMP/unknown.json" && ok "the unknown key is named in the refusal" || bad "the unknown key is not named"

echo "== image tag resolving to a different digest =="
"$PY" - "$PROFILE" "$TMP/bad_profile.json" <<'PY'
import json, sys
profile = json.load(open(sys.argv[1]))
profile["required_image_digest"] = "sha256:" + "0" * 64
json.dump(profile, open(sys.argv[2], "w"))
PY
"$PY" "$PROVISION" --check --profile "$TMP/bad_profile.json" "${ENV_ARGS[@]}" > "$TMP/tagmoved.json" 2>&1; rc=$?
[ "$rc" = 3 ] && ok "a moved/incorrect digest is refused (exit 3)" || bad "exit $rc, wanted 3"
grep -q "IMAGE_TAG_MOVED" "$TMP/tagmoved.json" && ok "verdict names IMAGE_TAG_MOVED" || bad "verdict not named"

echo "== --apply --dry-run on an absent container creates nothing =="
"$PY" "$PROVISION" --apply --dry-run --container tf-dry-run-probe "${ENV_ARGS[@]}" > "$TMP/dry.json" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "dry run exits 0" || bad "dry run exit $rc"
grep -q "create_argv" "$TMP/dry.json" && ok "the create argv is reported" || bad "no create argv"
if docker inspect tf-dry-run-probe >/dev/null 2>&1; then bad "dry run created a container"; else ok "nothing was created"; fi

echo
echo "passed: $pass, failed: $fail"
[ "$fail" -eq 0 ] && { echo "RESULT: PASS"; exit 0; }
echo "RESULT: FAIL"; exit 1
