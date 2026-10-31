# Adapter configuration

AgentGates bundles four independent, digest-pinned adapters. It runs fixed argv recipes from the selected source root. It does not import PiEngineeringHarness or require a Claude session.

| Adapter | Gate | Fixed operation | Inner / outer timeout |
| --- | --- | --- | --- |
| `ascendnpu-ir` | `ascend-build` | Ninja `bishengir-opt bishengir-compile install-bishengir-publish-products` | 840 / 900 s |
| `ascendnpu-ir` | `ascend-regression` | Ninja `check-bishengir`, after build | 1140 / 1200 s |
| `tilelang-hivm` | `tilelang-hivm-build` | Ninja `tilelang` | 840 / 900 s |
| `tilelang-hivm` | `tilelang-hivm-regression` | Two pinned host regression files, after build | 900 / 960 s |
| `tilelang-hivm-npu` | `tilelang-hivm-npu-sanity` | Fixed SIMD vector-add device test | 900 / 960 s |
| `tla-npu` | `tla-npu-e2e` | `test_tla.py`: TLA SIMD FP8 cast-back plus its plain-Triton reference, on one authorized device | 1200 / 1260 s |

The container timeout sends TERM then allows 15 seconds before KILL. The outer timeout remains larger. Nonzero exits, interruption and timeout never become PASS. The source must have a configured `build/build.ninja`; gates do not configure a fresh build tree.

## Machine configuration

Keep machine files outside the package and pass an absolute `--env-file`. Files contain strict `NAME=value` data, comments and blank lines. Balanced surrounding quotes are removed. Unknown keys, duplicate keys and malformed lines fail. The gate readers never execute the file or expand substitutions. CANN and Conda paths name locations inside the container. CANN is initialized before Conda; Conda setup and environment must be supplied together.

`--env-file` maps to `ASCENDNPU_ENV_FILE` for Ascend, `TILELANG_HIVM_ENV_FILE` for both TileLang adapters and `TLA_NPU_ENV_FILE` for the TLA adapter. Without an explicit selection, bridges look beside the packaged adapter template, retaining the legacy owning-checkout search for installed pack compatibility. They never search a user's home for machine files. An explicitly selected missing file fails.

Ascend permits `ASCENDNPU_CANN_SETUP`, `ASCENDNPU_CONDA_SH`, `ASCENDNPU_CONDA_ENV`, optional `ASCENDNPU_CCEC_BIN`, the container identity `ASCENDNPU_CONTAINER_NAME` / `_USER` / `_HOME`, `ASCENDNPU_CONTAINER_SOURCE`, and the build selection `ASCENDNPU_BUILD_SUBDIR` / `ASCENDNPU_NINJA_JOBS`. All of them are file keys: the launching environment is not a configuration source for any adapter. `ASCENDNPU_CCEC_BIN` is needed only by the from-zero branch, and `ASCENDNPU_CONTAINER_SOURCE` confirms the container path of the checkout rather than inventing one - the bind-mount mapping is still derived and compared. Only `ASCENDNPU_GATE_TIMEOUT` remains a launch value, because it bounds an invocation rather than describing the machine.

An export that the file does not define is refused, and an export that disagrees with the file is refused, so neither a stale shell nor a stale file can silently win. Sourcing `env.sh` and then running a gate is still supported, because the export then equals the file. AgentGates resolves the same file to take its `build:<dir>` lease, so a lock on one tree cannot cover a run writing another.

`ascend-build` selects its mode from the build directory inside the container: `build.ninja` present means incremental Ninja, neither `build.ninja` nor `CMakeCache.txt` means a from-zero full build through the pinned `scripts/full-build.sh` followed by product checks, and `CMakeCache.txt` without `build.ninja` is refused. The full build never deletes a tree, refuses `-r`/`--rebuild`, and never overwrites an existing `build.log`. `ascend-regression` runs an existing tree and carries no full-build branch.

TileLang's shared file permits:

- `TILELANG_HIVM_CONTAINER_NAME`, `TILELANG_HIVM_CONTAINER_USER`, `TILELANG_HIVM_CONTAINER_HOME`, `TILELANG_HIVM_CONTAINER_SOURCE`;
- `TILELANG_HIVM_CANN_SETUP`, `TILELANG_HIVM_CONDA_SH`, `TILELANG_HIVM_CONDA_ENV`, `TILELANG_HIVM_CONDA_ENVS_PATH`;
- `TILELANG_HIVM_HIVM_COMPILER`, `TILELANG_HIVM_HIVM_TARGET`;
- the build selection `TILELANG_HIVM_BUILD_SUBDIR`, `TILELANG_HIVM_NINJA_JOBS`;
- manual provisioning keys `TILELANG_HIVM_WHEEL_DIR`, `TILELANG_HIVM_PIP_INDEX`, `TILELANG_HIVM_PYTHON_VERSION`, `TILELANG_HIVM_SOURCE_ROOT`.

The file's container identity overrides stale launching exports, and `TILELANG_HIVM_CONTAINER_SOURCE` follows the same rule: it confirms the container path of the checkout when that differs from the derived bind-mount mapping, and because the mapping is still computed and compared, the key can confirm a path but never invent one. The NPU gate accepts the user key but runs as root, preserving the upstream `uda_occupy_dev_by_ns` workaround. A configured HIVM compiler must be an absolute executable container path. Its optional target can only be `Ascend950PR_9579`; compiler and target are explicitly forwarded as `HIVM_COMPILER` and `HIVM_TARGET`.

The TLA adapter has its own file and its own prefix. It permits `TLA_NPU_CONTAINER_NAME` / `_USER` / `_HOME` / `_SOURCE`, `TLA_NPU_CANN_SETUP`, `TLA_NPU_CONDA_SH`, `TLA_NPU_CONDA_ENV`, `TLA_NPU_CONDA_ENVS_PATH`, `TLA_NPU_PYTHON_VERSION`, and the provisioning keys `TLA_NPU_WHEEL_DIR`, `TLA_NPU_TRITON_WHEEL`, `TLA_NPU_PIP_INDEX`, `TLA_NPU_BUILD_SUBDIR`. It is a device adapter, so it also requires an explicitly authorized device through `--device-id` or `TLA_NPU_DEVICE_ID`, takes an `npu:<id>` lease on it, and resolves `TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE` when the local configuration sets the exclusivity policy - the engine forwards that policy into the variable each device adapter actually parses, rather than one hardcoded variable. `TLA_NPU_BUILD_SUBDIR` exists because the engine resolves a build directory for every adapter; this adapter writes no build tree, so it declares no job count. Unlike the TileLang gates, it does not require `build/build.ninja`: its source root is a package directory holding the test, and provenance is the source's filesystem hash.

The TLA adapter gates on what the process that ran the test actually measured: at least one tensor constructed on the device, a positive peak device allocation, and at least one kernel image in a fresh `TRITON_CACHE_DIR` (the Ascend backend emits `.npubin`; a `.o` is counted but not required, because requiring an artifact the backend never writes would be unsatisfiable rather than strict). The synchronization result, the two output markers and the script exit status are also gated, so a stub that prints the markers but never reaches the kernel still fails.

It does **not** claim to re-derive which physical board ran. The device UUID and name are recorded in the receipt but explicitly not gated: on this host both cards report the same UUID, and the visible-device echo is a value the adapter itself exported, so gating on either would be a tautology rather than evidence. `acl.rt.get_phy_dev_id_by_logic_dev_id` is likewise not used, because it answers from the host physical table regardless of which card `ASCEND_RT_VISIBLE_DEVICES` selected. Board-level identity cannot be observed from inside the container at all: the container's `npu-smi` fails to load `libc_sec.so`, so serials and PCIe addresses are host-side facts only. What the gate proves is that a compiled TLA kernel ran on the one card the operator named by `--device-id`, not that AgentGates independently re-identified that card.
## Commands and checks

```bash
agent-gates list --json
agent-gates doctor --adapter tilelang-hivm --source /absolute/source --env-file /absolute/machine.env --json
agent-gates run --adapter tilelang-hivm --source /absolute/source --env-file /absolute/machine.env --all --json
agent-gates verify --run-id RUN_ID --json
agent-gates report --run-id RUN_ID --json
```

Docker bind mounts determine the container source path. The most specific bind wins; missing or equally specific conflicting mappings fail. The bridge compares host/container source device and inode before execution. Core evidence also binds source identity, adapter manifest and script digests, executable, configuration, process logs and discovered artifacts. Verification determines whether a recorded result is still current.

TileLang regression runs exactly `testing/ascend/target/test_tilelang_ascend_target.py` and `testing/ascend/cache/test_tilelang_ascend_binary_cache.py`. JUnit supplies collected, passed and skipped counts. A positive collected count, at least one non-skipped test, native-library presence and host/container module/native SHA256 agreement are required. These are host regression checks; they do not establish complete HIVM code generation or device correctness.

The separate NPU adapter requires an explicitly allocated device, using CLI `--device-id ASSIGNED_ID` or `TILELANG_HIVM_DEVICE_ID`. Neither an idle card nor visibility authorizes selecting it. On top of that declaration the Adapter performs a read-only Host probe with `npu-smi info` and refuses to dispatch when the card already holds any process the Gate did not start, or when the driver cannot be read at all: a missing binary, a non-zero exit, a timeout or an unrecognised answer is treated as unknown occupancy, not as free. The probe maps the authorized physical id through the `npu-smi` mapping table rather than assuming NPU ID equals physical id, runs both in `--check-config` and again immediately before the container command, and reports only the holder count. It never selects, resets or retries a card, and it cannot see a job that starts after the probe; shared-machine scheduling remains the operator's responsibility. It requires measured tensor movement, numerical assertions, positive device allocation, matching authorized physical-device mappings before and after, collected passing tests and successful synchronization. The scope is fixed SIMD vector-add sanity. Running `--all` on the base adapter never invokes this gate.

## Manual support and provenance

Full-build, initialization, Python provisioning and Docker support files are retained from the source adapter. Gates never call the full-build/provisioning/container-creation helpers. Container-creation configuration is trusted Bash and is sourced by that manual helper; it is distinct from the strict gate data file. Review it before deliberate use. Upstream deployment defaults include privileged execution. Device visibility has no default: an omitted or empty `ASCEND_VISIBLE_DEVICES` means no runtime filtering, so name an explicitly reviewed allocation when provisioning. No deployment helper was run during offline adapter validation.

Migrated scripts originate in PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`. Shell helpers, Dockerfiles and entrypoints are copied without changing compiler commands. Bridge changes add packaged sibling configuration discovery and an Ascend host-only `--check-config`; all affected manifest pins are updated. Obsolete Harness installation and orchestration instructions were replaced. No upstream root license was found in the audited checkout, so this migration records provenance without inventing a license grant. Machine `env.sh`, `docker/config.env`, build outputs and real run logs are excluded from the distribution.
