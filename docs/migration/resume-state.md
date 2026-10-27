# Pause checkpoint (superseded)

This file recorded a pause on 2026-10-27 at the user's request. **Work resumed afterwards; this checkpoint is historical.** The current state is in the [execution ledger](progress.md) and the [final report](final-report.md).

Kept because it records what the pause preserved and what was still open at that moment.

## State at pause

- Branch `migration/p0-p2`, last commit `3994c69`. Core package, adapters, plugin, tests and most docs were untracked work in progress; nothing was stashed or discarded.
- No push had been performed. GitHub PRIVATE visibility was unproven at that point.
- `.venv/`, `.evidence/`, raw logs and local derived machine configuration were ignored and preserved.

## Open items at pause, and their outcome

| Open at pause | Outcome |
| --- | --- |
| NPU occupancy probe missing | Implemented in `npu_gate.py`: read-only Host `npu-smi` probe in `--check-config` and again before dispatch, fail-closed on every unknown; 10 offline tests; verified against both real cards |
| Adapter scripts edited after pause without repinning (15 failures) | Manifests repinned; suite green |
| Exact-model Opus retry running, no verdict | Never produced a completed review; recorded as an evidence gap, not claimed |
| P1 real CLI acceptance not launched | Completed for all three adapters, plus two explicit-device occupancy runs |
| Isolated wheel proof not executed | Executed, PASS |
| P2 fresh plugin session not executed | Executed, exit 0; worktree exercises honestly blocked on missing builds |
| GitHub PRIVATE unproven | Proven private on four independent facts (see ledger) |
| `docs/testing.md`, `p1-equivalence.md`, `p2-validation.md`, `final-report.md` | Written |

## What the pause got right and wrong

It correctly refused to claim completion while the NPU gate and repo visibility were unverified, and it preserved the working tree without destructive cleanup. It also correctly recorded that the 12 TileLang native `.so` hashes were not persisted in P0 — that gap was not retroactively filled.

It did not record that the NPU adapter had no occupancy check at all. That was found after resuming and fixed; it is the one behavioural defect the pause state would have shipped.
