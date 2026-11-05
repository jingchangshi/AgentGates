# Shared device-lock incident — 2026-11-05

Deleting a shared lock file has side effects, so the action, the evidence behind it
and the standing procedure are recorded here rather than only in a commit message.

## 1. What was found

    $ ls -la /home/s00653124/.tf_device0.lock
    -rw-r--r-- 1 root root 0 Nov  4 20:25 /home/s00653124/.tf_device0.lock

- **Observed**: 2026-11-05 ~13:1x while bringing up the team lock lease (P0-E).
- **Owner**: `root:root`, mode `0644`.
- **Size**: 0 bytes — it carried no holder record.
- **Session account**: uid 1071, in the `docker` group, **not** root (`sudo` needs a
  password that this account does not have).
- **Effect**: the lease could not even `open(..., "a+")` the file, so every device run
  in the CPU-side tests failed. It surfaced as `device_policy.py` reporting
  `TypeError: object of type 'NoneType' has no len()` and `negatives.sh` failing the
  busy/idle case — i.e. as misleading downstream symptoms, not as a permission error.

## 2. Why it is judged a LEFTOVER, not a live lock

1. **Zero length.** The lease writes a JSON holder record (pid, acquisition time,
   nonce) and truncates on release. An empty file is what remains after a truncating
   release, never what a live holder leaves behind.
2. **No holder record and therefore no active holder claim.** A genuine in-flight
   holder would have had a parseable record naming a live pid.
3. **No process of ours was running a device job** at the time; no team job was in
   flight.
4. **Ownership is root**, and this team's jobs run as uid 1071. A file root created
   on 11-04 at 20:25 predates the lease implementation (written today), so it was left
   by an earlier **root** device run — consistent with the documented root channel
   (`docker exec -u root tf-ascend950pr`).
5. **`flock` was free**: the lock could be acquired by a kernel flock test, so no
   process held it; only the file's *ownership* blocked opening it.

Conclusion: a stale root-owned artifact with no lock held, blocking the team account.

## 3. Action taken, and its blast radius

    $ rm -f /home/s00653124/.tf_device0.lock     # empty regular file, no holder record
    removed

- **What was removed**: one 0-byte regular file. No data was lost; a live lock cannot
  be removed this way without its holder's cooperation, and none existed.
- **Who was affected**: only this team's own lock path. Other tenants were never
  coordinated by this file (the lock is *team* serialisation), so no foreign job could
  be affected.
- **Not done, deliberately**: no `chown`, no mode change, no `sudo`, no container
  restart, no NPU reset. Those would have wider effects than deleting a file whose
  purpose is to be created on demand.
- **Verification after removal**: the lock is recreated by the next run and the
  permissions are those of the creating account.

## 4. Code change that came out of it

A lock that cannot be opened is now a **named refusal**, not a raw traceback:

    DEVICE_LOCK_NOT_WRITABLE: cannot open <path> (<errno>). The lock is a plain file in
    the team's home; a root-owned leftover from an earlier run makes it unusable by the
    team account. Remove it or point TF_DEVICE_LOCK_PATH at a writable location. This is
    an environment fault, not device busyness.

The message matters as much as the code: a `PermissionError` traceback reads like a bug
in the Gate, while the real cause is a leftover file, and the two need different fixes.

## 5. Standing procedure when a leftover lock is met

1. **Do not** loop, sleep, or retry — the lease already bounds the wait.
2. Read the refusal: `DEVICE_LOCK_NOT_WRITABLE` means the FILE is unusable (ownership
   or permissions); `DEVICE_LOCK_HELD_BY_TEAM` means another of our jobs is inside the
   payload and the bounded wait expired.
3. For the former: `ls -la` the path. If it is root-owned and 0 bytes with no holder
   record, it is a leftover — remove it (as above, recording the action) **or** set
   `TF_DEVICE_LOCK_PATH` to a writable path for the run.
4. For the latter: identify the holding pid from the message; do not kill another
   user's process. Wait for it, or coordinate.
5. **Never** make the lock world-writable or root-owned-on-purpose to "fix" it: that
   trades a loud failure for a silent one.

## 6. Environment isolation used by the tests

`tests/test_e2e_chain.py` points `DEVICE_LOCK_PATH` at a private per-test directory, so
the suite measures lease BEHAVIOUR and is not sensitive to any real leftover. The
end-to-end tests that do exercise the real path are the ones that must run in the real
environment.

---

## 7. Correction: the canonical path, and what each date actually was

An earlier draft of this note described one incident. There are two, on different
paths, and conflating them produced a wrong conclusion.

| path | state | meaning |
|---|---|---|
| `/home/.tf_device0.lock` | **root:root, mode 0666, mtime 11-05 10:21** | THE canonical team lock, recreated by the container's root side. Mode 0666 means every team account can open it, so it is USABLE. |
| `/home/s00653124/.tf_device0.lock` | uid 1071, mode 0664, mtime 13:24 | An artifact of running tests with `TF_DEVICE_LOCK_PATH` pointed at a private path. NOT the team lock. |

The 11-04 event was: `/home/s00653124/.tf_device0.lock` existed as **root:root mode
0600**, i.e. **not writable** by uid 1071, and the default path at that time resolved
under the home directory. That is what produced `DEVICE_LOCK_NOT_WRITABLE` and the
misleading downstream `TypeError` in `device_policy.py`.

### What changed as a result

- The default is now the canonical `/home/.tf_device0.lock` and must stay that way: a
  per-user path under the home directory would silently fail to serialise two team
  members whose accounts differ, which defeats the purpose of a *team* lock.
- `TF_DEVICE_LOCK_PATH` remains an explicit override for tests and isolation ONLY. When
  in effect the result record sets `lock.canonical = false` with a note, so a
  private-lock run can never be cited as evidence that team jobs serialise.
- Two-process serialisation is tested against the CANONICAL path; the in-process lease
  cases use a private path and claim only the mechanism.
- A lock that cannot be opened is a named `DEVICE_LOCK_NOT_WRITABLE`, not a traceback.

### Timeline

    11-04 20:25  root:root 0600 file appears (earlier root/container run)
    11-05 10:21  canonical /home/.tf_device0.lock recreated root:root 0666 (usable)
    11-05 ~13:10 DEVICE_LOCK_NOT_WRITABLE observed; causal chain traced
    11-05 13:24  stale home-directory file removed (0 bytes, no holder, lock free)
    11-05 13:30  default corrected to canonical; private override self-identifies
