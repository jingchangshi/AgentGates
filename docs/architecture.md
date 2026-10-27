# Architecture

AgentGates owns deterministic execution and evidence. Compiler semantic analysis belongs in the compiler; planning, editing and review belong in the calling developer or agent. No Harness task store, supervisor communication bridge, model router or agent orchestration is included.

`registry.py` discovers bundled adapters. `manifest.py` validates schema version 1, unique identifiers, gate dependencies, fixed argv/CWD, timeouts, and pinned file digests. `{pack}` expansion can only address pinned files inside the adapter; symlink and traversal escapes are rejected. Gates are topologically ordered within one adapter.

`engine.py` resolves runtime configuration, acquires source/build/device leases, executes preflight checks, captures source identity before and after execution, evaluates gates, and publishes evidence. `executor.py` dispatches fixed argv using `shell=False`. A Linux subprocess supervisor retains resource leases, detects loss of its parent, and terminates/reaps its owned descendants. Adapter Docker bridges retain their own container-side timeout; the existing shared container is never stopped for cleanup. Container-side work cannot be assumed dead merely because a Docker client died; the adapter timeout bounds that separate process domain.

`evaluator.py` compares required observations against real stdout with type-sensitive checks. Nonzero exit, timeout, interruption, malformed observations, missing fields, zero tests and all-skipped tests cannot produce PASS. Exit-code-only build gates accept non-JSON build logs while structured-observation gates require a JSON object.

`source_identity.py` hashes committed HEAD and tracked/dirty/untracked source bytes using Git's nonignored file enumeration, with special handling for symlinks and submodules. Ignored build artifacts are bound separately. Source trees containing unsupported path boundaries fail closed rather than silently accepting incomplete identity.

`evidence.py` records and verifies local receipts, logs, process results, adapter identity, source identity, configuration-file identity, and product hashes. Validation replays the acceptance conditions from the manifest. It does not reuse historical PASS for a new run. See [evidence trust and limits](evidence.md).

`locking.py` provides cross-process advisory leases for AgentGates runs. They do not coordinate arbitrary other users' commands. The `tilelang-hivm-npu` Adapter adds a read-only Host `npu-smi` occupancy probe that blocks dispatch when the authorized card already holds another process or cannot be read; it is a snapshot, not a reservation, so a job starting after the probe still races it. Device assignment remains external and a shared machine requires scheduling discipline.

The optional Claude Code plugin contains CLI-only skills and an opt-in task-completion receipt check. It never supplies an alternate gate evaluator or automatically launches hardware tests for unrelated tasks.
