# megakernel-npu

Host-local AgentGates adapter for `Triton-distributed-ascend` (`feature/megakernel`).

## Gates

| Gate ID | Operation | CANN | NPU Lease |
| --- | --- | --- | --- |
| `megakernel-npu-build` | Incrementally syncs `--source` into `MEGAKERNEL_NPU_WORK_ROOT/stage` (keeping `--source` read-only), applies `triton-ascend` patches offline in staging, builds `triton_dist` with `<= 16` cores against `MEGAKERNEL_NPU_LLVM_SYSPATH`, installs the wheel into `megakernel-npu`, and verifies `libtriton.so`, opt tools, MegaKernel bitcode (`megakernel_op.*.c310.bc`), and `ModelBuilder` | Sourced (`MEGAKERNEL_NPU_CANN_SETUP`) | None (`device: none`) |
| `megakernel-npu-cpu` | Runs Host/CPU `pytest` suite (`TORCH_DEVICE_BACKEND_AUTOLOAD=0`, without CANN) with JUnit XML verification | Not sourced | None (`device: none`) |
| `megakernel-npu-device` | Depends on `megakernel-npu-cpu`; acquires `deviceWindow` (`waitSeconds: 60`), sources CANN + `megakernel-npu`, runs NPU & MegaKernel verification probe (`torch`, `torch_npu`, `triton_dist`, `ModelBuilder`, FP32 vector add, BF16 matmul, synchronization, device memory), and optionally executes `MEGAKERNEL_NPU_DEVICE_TESTS` | Sourced | Windowed (`deviceWindow`: 60 s) |

## Machine Configuration (`env.sh`)

Copy `env.example.sh` to git-ignored `env.sh` beside it and set host paths:

```bash
cd agent_gates/adapters/megakernel-npu
cp env.example.sh env.sh
chmod 600 env.sh
```

- `MEGAKERNEL_NPU_NINJA_JOBS` is strictly validated in `1..16` (defaults to `16`).
- `MEGAKERNEL_NPU_WORK_ROOT` must sit outside `--source` so `setup.py` patching and CMake/Ninja artifacts never mutate the checked-out repository (`sourceWrite: false`).
- `MEGAKERNEL_NPU_DEVICE_ID` and `MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE` are launch values (never written to `env.sh`).

## Commands

```bash
agent-gates doctor --adapter megakernel-npu --source /abs/path/to/Triton-distributed-ascend --device-id 0 --json
agent-gates run --adapter megakernel-npu --source /abs/path/to/Triton-distributed-ascend --gate megakernel-npu-build --json
agent-gates run --adapter megakernel-npu --source /abs/path/to/Triton-distributed-ascend --gate megakernel-npu-cpu --gate megakernel-npu-device --device-id 0 --json
```

## GLM5.2 single-card golden validation (local extension)

`scripts/glm52_single_card.py` runs the model-level BlockRT/TD golden pipeline
(`python/triton_dist/mega_kernel_ascend/test/models/test_glm5_2_golden_pipeline.py`).
It is a local adapter extension: it does not modify the digest-pinned manifest or
the three gates above. Machine paths live in `glm52.env` (git-ignored, beside
`env.sh`); CANN, Conda and Python default to `MEGAKERNEL_NPU_CANN_SETUP` /
`_CONDA_ENV` / `_PYTHON` from `env.sh`. `GLM52_BUILD_JOBS` is strictly capped at 16.
`GLM52_BISHENGIR_BIN` must name a directory containing a matched
`bishengir-opt` / `bishengir-compile` pair (CANN 9.3 can ship dangling symlinks);
it is prepended to `PATH` and exported as `TRITON_NPU_COMPILER_PATH`.

```bash
cd <repo-root>/agent_gates/adapters/megakernel-npu
cp glm52.env.example glm52.env
chmod 600 glm52.env
# fill in the host paths, then:
python3 scripts/glm52_single_card.py config      # resolved paths, no execution
python3 scripts/glm52_single_card.py prepare     # export pinned reference source
python3 scripts/glm52_single_card.py build       # reference Dump + AscendC tiling ext, <=16 jobs
python3 scripts/glm52_single_card.py build-tiling # AscendC host tiling ext only
python3 scripts/glm52_single_card.py preflight   # Dump flag, APIs, tiling ext, manager ABI
python3 scripts/glm52_single_card.py run --layers 1 --device-id 0
python3 scripts/glm52_single_card.py all --device-id 0   # layers 1, 2, 4
```

The tiling extension is compiled from the live TD source in an overlay tree; its
`3rdparty/ascendc-ops` is linked to the reference checkout's
`examples/operators` bundle, so the TD repository stays read-only.

For the GLM5.2 model the TD build must also include the AscendC device kernels:
set `MEGAKERNEL_NPU_ENABLE_ASCENDC_OPS=ON` in `env.sh` and create the
untracked symlinks `<TD>/3rdparty/ascendc-ops -> <reference>/examples/operators`,
`<TD>/3rdparty/catlass -> <reference>/3rdparty/catlass`, and
`<TD>/3rdparty/catlass_a5 -> <reference>/3rdparty/catlass_a5` before running
`megakernel-npu-build`. Without them the dispatch link fails on
`_mlir_ciface_dtile_ascendc_*` symbols or the op build cannot find
`catlass/catlass.hpp`, and the installed `megakernel_op.*.c310.bc` files stay tiny.

The runner never selects a card: `--device-id` is required and must be
explicitly authorized. It does not run `install.sh`; it verifies that the
built `libaicpu_manager.so` exports the same symbol set as the already-installed
`Ascend-aicpu_pk.tar.gz` and fails closed otherwise. A run passes only when the
pipeline exits 0, the log contains `[ALLPASS]`, and the top-level
`comparison.json` has `passed: true`; `--generate-only` is not a pass. Logs go
to `GLM52_LOG_ROOT`; Golden and run outputs go under `GLM52_GOLDEN_ROOT`.
