# Migration execution ledger

Execution date: 2026-10-27. Goal: local P0–P2 extraction of deterministic compiler gates; no compiler analysis framework.

## Team and ownership

Native Claude Code 2.1.296 Agent tool started three named teammates with native session team metadata present. Lead runtime reports codex/gpt-6.1-sol; it is not represented as Opus. Requested model selections: core-implementer Sonnet; adapter-plugin Sonnet; independent-reviewer Opus. Actual implementer response model fields both report `gpt-6.1-sol`. The Opus reviewer transcript contains an initial `claude-opus-5-5` response, safety-classifier notices, then repeated synthetic `model_not_found` errors naming `claude-opus-5`. It produced no audit deliverables and was stopped after the user flagged availability. A replacement `independent-validator` was started on the configured default model for local correctness audit and offline validation. This is a separate reviewer instance, not an implementer's self-review; no claim of successful Opus review is made. Core owns Python core/project configuration; adapter teammate owns adapter data/plugin; reviewer owns equivalence fixtures/audit. Lead owns integration and actual machine test scheduling. No simultaneous writes to the same file or build tree.

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
- gh absent; no GH_TOKEN/GITHUB_TOKEN or usable HTTPS credential helper credential. Public API lookup 404; authenticated attempt without credential 401. SSH remote readable but that does not prove visibility. Push forbidden until private verified.
- A5 model Ascend950PR; driver/npu-smi 25.7.rc1. Device 0 has a running Python process; device 1 currently idle. No explicit assigned device ID observed. Do not choose a device merely because idle.
- Existing user TileLang container has /home and /data bind mounts, permitting new worktrees in those trees in principle. No images rebuilt or shared containers stopped.

Local sensitive logs/configuration are stored only under ignored `.evidence/`. No machine env.sh, credentials, full source trees or binary artifacts enter Git.
