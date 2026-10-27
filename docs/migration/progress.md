# Migration execution ledger

Execution date: 2026-10-27. Goal: local P0–P2 extraction of deterministic compiler gates; no compiler analysis framework.

**Status: P0–P2 delivered locally on branch `migration/p0-p2`; remote delivery is the last step and is reported at the end of this file.** The pause recorded below is historical; the resume found and fixed one behavioural defect (the NPU adapter had no device occupancy check), then completed the remaining validation. Current results are in [P1 equivalence](p1-equivalence.md), [P2 validation](p2-validation.md) and the [final report](final-report.md). The [pause checkpoint](resume-state.md) is retained as history only.

## Team and ownership

Native Claude Code 2.1.296 Agent tool started three named teammates with native session team metadata present. Lead runtime reports codex/gpt-6.1-sol; it is not represented as Opus. Requested model selections: core-implementer Sonnet; adapter-plugin Sonnet; independent-reviewer Opus. Actual implementer response model fields both report `gpt-6.1-sol`. The Opus reviewer transcript contains an initial `claude-opus-5-5` response, safety-classifier notices, then repeated synthetic `model_not_found` errors naming `claude-opus-5`. It produced no audit deliverables and was stopped after the user flagged availability. A replacement `independent-validator` was started on the configured default model for local correctness audit and offline validation. This is a separate reviewer instance, not an implementer's self-review; no claim of successful Opus review is made. Core owns Python core/project configuration; adapter teammate owns adapter data/plugin; reviewer owns equivalence fixtures/audit. Lead owns integration and actual machine test scheduling. No simultaneous writes to the same file or build tree.

### Explicit Opus retry

The user identified the configured `ANTHROPIC_DEFAULT_OPUS_MODEL=claude/claude-opus-5-5` and requested a retry using that exact name. Lead confirmed the variable and launched a fresh native Claude Code review session with `--model claude/claude-opus-5-5` (the Agent tool model enum cannot express a full provider model name). Session `333e2661-73d9-4bed-b7f8-eaaca366281a` initialization records that exact model, with local transcript `.evidence/opus-review-retry.jsonl`. Subsequent actual assistant responses reported `claude-opus-5-5` and began the read-only review, confirming inference worked on this exact-name retry. That session was stopped before producing any verdict; see "Independent review" below. The default-model validator's independent results remain preserved.

## Dependencies

1. Parallel P0 audits + lead machine inventory / GitHub privacy verification.
2. Original fixed recipes baseline, only after safe configuration/resource checks.
3. Core and adapter implementation after interface coordination; reviewer fixtures independently.
4. Offline tests, isolated installation and independent review.
5. Sequential new CLI real machine gates; explicit authorization required for particular NPU ID.
6. Fresh plugin load/skills/hook + isolated compiler development worktrees.
7. Final evidence summaries, commits and safe remote delivery only if privacy proven.

## Initial facts and constraints

- Target empty unborn main; no tracked initial files.
- Source Harness local HEAD and remote master/HEAD: `9ca04ba6814154a778ecf30168c793d73352e643`, matching requested reference; clean.
- AscendNPU-IR HEAD `bfefe18011152cca671e7e6ddf03793afbacaa2a`, branch `feat/tilelang-hivm-late-pipeline`, clean.
- TileLang HEAD `6fe6f09182c854e680a5488f4d8461fe5d5a76a6`, branch `feat/hivm-backend-poc`, five pre-existing tracked modifications. Preserve.
- Python 3.11.14, Node 24.20.0, Git 2.55.0, Docker 28.1.1, tmux 3.7c.
- gh absent; no GH_TOKEN/GITHUB_TOKEN or usable HTTPS credential helper credential. SSH key authentication works for the account. Visibility was later proven PRIVATE with a control repository and the account's complete public listing; see "GitHub visibility" below.
- A5 model Ascend950PR; driver/npu-smi 25.7.rc1. Two device IDs are visible. No explicit assigned device ID was observed, and an idle card was never treated as authorization. Occupancy readings across the day disagreed (inventory: none visible; ledger: device 0 held; final probes: device 0 held by 2 and device 1 by 5 processes not started by any Gate), which is why occupancy is probed per run rather than recorded as a fact.
- Existing user TileLang container has /home and /data bind mounts, permitting new worktrees in those trees in principle. No images rebuilt or shared containers stopped.

Local sensitive logs/configuration are stored only under ignored `.evidence/`. No machine env.sh, credentials, full source trees or binary artifacts enter Git.

## Resume: defect found and fixed

The resume found one behavioural defect that the paused state did not record: **the NPU adapter had no device occupancy check at all.** It required an explicit `TILELANG_HIVM_DEVICE_ID` and mapped the logical to the physical device, but never asked whether that card was already in use. On this machine both cards are held by other users' jobs, so an operator who guessed a device id would have shared a card with a running job.

Fix, in `agent_gates/adapters/tilelang-hivm-npu/scripts/npu_gate.py`:

- `npu_smi()` runs only `npu-smi info` subcommands on the Host, with a 60 s bound. A missing binary, non-zero exit, timeout or `OSError` is a `BridgeError`, never "free".
- `npu_id_of_physical()` maps the authorized physical id through the `npu-smi -m` table instead of assuming NPU ID equals physical id; an absent or ambiguous row is refused.
- `processes_on()` requires the answer to name the queried NPU ID; anything else is unknown occupancy.
- `require_device_free()` reports only the holder count, never other users' PIDs or command lines.
- It runs in `--check-config` and again immediately before `docker exec`, because the configuration check may have run minutes earlier.

The container's own `npu-smi` is unusable in the current image (`libc_sec.so` fails to load), which is why the probe runs on the Host. Ten offline tests cover the busy, unreadable, missing-binary, hanging, missing-mapping-row, recheck and read-only-only cases. Real-machine confirmation: device 0 and device 1 both refused dispatch, with the reason recorded on the gate as `configReason`.

`engine.py` also gained `config_reason()`, which puts one bounded line of the Adapter's own stderr on the blocked gate. The machine-readable failure marker `CONFIG_CHECK_FAILED` is unchanged, so existing tests and consumers are unaffected.

## Final validation

| Item | Result |
| --- | --- |
| Offline suite | 441 passed, 88 subtests passed, exit 0 (`.evidence/final-pytest.log`; 439 before the two packaging tests were added) |
| Isolated wheel install | PASS (pinned files present, no machine config, no Harness/Claude/Anthropic import) |
| P1 `ascendnpu-ir` | doctor + run PASS, `c1610b674026bc2156c1a3ccedb48ee0`, verify `valid: true` |
| P1 `tilelang-hivm` | doctor + run PASS, `181cad5aa72c8893043f4a53f7b2e79b`, verify `valid: true` |
| P1 `tilelang-hivm-npu` | BLOCKED_ENV, no assignment; verify `valid: false` |
| Occupancy probe, device 0 / 1 | BLOCKED_ENV `a0c167250ab7c79a1e286ddf4e8cd7c2` / `a3428763b378f1b6a0297b4bf42c8c97` |
| `claude plugin validate` | passes, no warnings |
| P2 fresh plugin session | exit 0, 243.3 s, three skills discovered, plugin-initiated TileLang PASS `74e428e9818aa75ad7fb1b95f20032bc` |
| P2 worktree exercises | both FAIL honestly on missing worktree builds; not redirected to the original checkouts |
| GitHub visibility | PRIVATE, proven on four independent facts |

## GitHub visibility

Four facts, each with a control, recorded in ignored `.evidence/private-check.log`:

1. `git ls-remote --heads origin` over SSH succeeds, exit 0, zero refs — the repository exists and this account is authorized.
2. The same SSH identity against `jingchangshi/AgentGates-does-not-exist-xyz` returns exit 128 "Repository not found", so (1) is a real answer and not a generic connection result.
3. The anonymous GitHub API returns 404 for the target while `api.github.com/repos/git/git` returns 200 on the same endpoint at the same moment — a denial, not a network fault.
4. The account's complete public repository listing (102 repositories across 3 pages) does not include AgentGates.

No credential value, token or private key was read or written by this check.

## Independent review

No completed Opus review exists. The first reviewer failed with `model_not_found` for the `claude-opus-5` alias; the exact-name retry (`claude/claude-opus-5-5`, session `333e2661-73d9-4bed-b7f8-eaaca366281a`) initialised with the correct model but produced no verdict before being stopped. The P2 session ran on `group/auto-deepseek-v4-1-flash`, which is not Opus. A read-only independent review was run after resuming; its findings and dispositions are recorded in the final report.
