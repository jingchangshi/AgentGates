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
