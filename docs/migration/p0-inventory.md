# P0 inventory

Inventory date: 2026-10-27. Machine-specific raw records are local under ignored `.evidence/`; this document contains only a limited summary.

| Checkout | HEAD | State |
| --- | --- | --- |
| AgentGates | unborn main at start | Existing untracked ignore file and execution ledger preserved |
| PiEngineeringHarness | `9ca04ba6814154a778ecf30168c793d73352e643` | Clean; remote HEAD independently queried and equal to requested baseline |
| AscendNPU-IR | `bfefe18011152cca671e7e6ddf03793afbacaa2a` | Clean at inventory |
| TileLang | `6fe6f09182c854e680a5488f4d8461fe5d5a76a6` | Five existing tracked modifications; preserve them |

The existing TileLang changes are in `src/ascend/codegen/hivm_codegen.cc`, `src/ascend/codegen/rt_mod_ascend_hivm.cc`, `tilelang/ascend/backend.py`, `tilelang/ascend/codegen.py`, and `tilelang/contrib/hivm.py`.

Versions: Claude Code 2.1.296, Python 3.11.14, Node 24.20.0, Git 2.55.0, Docker 28.1.1, tmux 3.7c. Lead runtime identifies itself as codex/gpt-6.1-sol; no claim that the lead is Opus is made. Three native named teammates were launched with requested models Sonnet, Sonnet and Opus.

## Environment and device boundaries

The existing user build container is running. Its image content identity and bind mounts were inspected; `/home` and `/data` are mounted, allowing a future isolated worktree under those roots without remounting. No container was created, stopped, or rebuilt. The actual device model is Ascend950PR (A5), driver/npu-smi 25.7.rc1. Two device IDs are visible. At the current check neither lists a running process, but availability changes and idle state does not constitute assignment. No task-specific authorized device ID was present in the launching environment. NPU execution remains blocked until assignment is established.

TileLang's existing machine file was parsed as data, using only adapter allowlisted keys. It specifies existing CANN 9.2.0 setup, an existing shared Conda installation and user environment, existing container identity, an existing installed `bishengir-compile`, and `Ascend950PR_9579` as target. The setup files and compiler executable exist inside the container. No private configuration file is committed. `TILELANG_HIVM_ENV_FILE`, `ASCENDNPU_ENV_FILE`, and `TILELANG_HIVM_DEVICE_ID` were unset in the launching environment. No Ascend adapter machine file was present beside its template; a local derived configuration may reuse verified existing toolchain/container values without modifying original files.

Both source trees have existing `build/build.ninja`. Ascend's configured C/C++ compiler is Clang 15.0.6; TileLang uses the container system C/C++ compiler. Both builds reference the existing user Python 3.11 Conda environment. Ascend build and installed compiler executables exist. Actual hashes and executed Ninja targets are recorded in baseline evidence.

## Remote delivery prerequisite

`gh` is not installed. No `GH_TOKEN`/`GITHUB_TOKEN` or usable HTTPS credential-helper password was found. Anonymous GitHub repository API lookup returned 404; the SSH remote is readable and empty, which cannot prove visibility. Consequently private visibility cannot currently be established and all pushes are prohibited. Local development and commits remain authorized. Do not infer private status from anonymous 404.

## Source and installed pack audit

The source manifests and their pinned files are the trusted baseline; installed-pack comparison and detailed semantic findings are recorded in `p0-audit.md` and `gate-semantics.md`. Migration changes must preserve fixed argv, timeouts, observations, and adapter script digest checks. Any script change must be recorded with updated manifest digests and regression tests.
