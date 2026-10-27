# Evidence and trust

Each execution receives a random run ID and its own directory below the evidence store's `runs/`. The receipt identifies adapter manifest and pinned scripts, selected gates and dependency order, original argv/CWD, committed HEAD and source-byte snapshot, configuration-file digests, times, process exit/timeout/interruption, stdout/stderr hashes, parsed machine observations, acceptance checks, artifact identities, status and failures. Logs and terminal process records remain separate files and are hashed by reference.

The default store is `~/.local/state/agent-gates`; a command may choose `--evidence-root`. Keep machine logs/configuration out of Git. This repository ignores `.evidence/`, `.agent-gates/`, machine env files, environments and build products. Share only reviewed, redacted summaries.

Receipts use a local per-user HMAC key. This detects accidental corruption and receipt edits by a party that lacks that key. It cannot resist an agent or process with the same OS-user privileges: that process can read the key or modify the program. The trust label is `local-user-integrity`, not a sandboxed or externally signed attestation. Stronger isolation requires a separately controlled signer and execution boundary.

The receipt also binds the adapter's own environment assignments and the inherited toolchain variables (`PATH`, `PYTHONPATH`, `PYTHONHOME`, `LD_PRELOAD`, `LD_LIBRARY_PATH`, `HOME`, `SHELL`) as two SHA-256 digests. Values are never stored. They give two runs under different toolchains distinct configuration identities, and the run compares both before and after each gate. `verify` deliberately re-derives neither: the verifying shell's environment is not the running shell's, so a different `PATH` would otherwise mark every sound receipt `STALE`. What `verify` re-checks from `configuration` is the referenced machine files.

Verification checks HMAC, current manifest/script identity, source snapshot, configuration files, recorded argv/CWD and gate dependencies, logs and terminal process records, executable and product references, and reevaluates PASS observations from stored stdout. Report uses this verification result as its current status and preserves the original recorded status separately. The tool does not use a previous PASS as the result of a new execution.

| Status | Meaning |
| --- | --- |
| `PASS` | All selected gates and dependencies passed the recorded deterministic rules |
| `FAIL` | An executed command or observation failed its conditions |
| `BLOCKED_ENV` | Required configuration/resource/environment prerequisite was unavailable |
| `NOT_RUN` | A gate was not executed, for example because a dependency did not pass |
| `INVALID_EVIDENCE` | Receipt, logs, identity bindings or manifest evidence cannot be validated |
| `STALE` | Evidence no longer matches the current source/configuration/product identity |

Source snapshots include Git-listed tracked and nonignored untracked bytes, file modes, symlink text and recursively measured submodules, plus Git index identities. A symlink's target bytes are not followed by the source snapshot; relevant executable/product targets are separately resolved and hashed. Git-ignored files are excluded from source identity; relevant compiled products are bound separately. This is a declared scope, not an assurance that arbitrary ignored files or external symlink targets used as compiler inputs are covered. Avoid hiding test or source inputs behind ignore rules or external links.

Local file identity does not attest all machine state: drivers, all transitive system libraries, external processes and resource assignment can change. Adapter module/native hashes attest the measured products; they do not establish full compiler semantic correctness. The basic NPU SIMD sanity test has limited coverage and must not be described as full HIVM backend E2E validation.

A Docker client and the container process belong to separate process domains. Host process-group cleanup cannot be treated as proof that container work stopped instantly. The pinned bridge enforces a container-side timeout with a kill grace period, and the recipe timeout is longer. Shared containers are never stopped to clean up a gate.
