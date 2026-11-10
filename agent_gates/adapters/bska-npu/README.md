# bska-npu Adapter

Runs and verifies the BiShengKernelAgent (helix) NPU runtime and the vendored
official CANNBench evaluator on the host, against one conda environment and one
authorized card. No container.

## What it proves

Two recipes, in order:

* `bska-npu.smoke` — CANN is sourced; the configured interpreter imports torch,
  torch_npu, triton and torchvision; a real Triton vector-add runs on the
  authorized card and matches its Torch reference; the CANNBench native
  extension loads *and executes* on that card; node, claude and helix are
  available.
* `bska-npu.cannbench` — the vendored official evaluator scores one operator and
  produces a report the project would accept: every case passed and every case
  has a valid timing. Depends on `bska-npu.smoke`.

Neither step claims the other's result. A green smoke run says the runtime
works; it says nothing about whether the evaluator can produce a score.

## Why no container

An earlier revision entered a Docker container as root. Two defects came out of
that, and both were silent:

* the container's login shell exported `ASCEND_RT_VISIBLE_DEVICES`, overriding
  the card the Gate was told to use;
* running as root changed `HOME`, so the interpreter resolved the
  environment-site Torch (2.10) instead of the user-site Torch (2.13) that the
  native extension had been compiled against.

Under the mismatch every import still succeeded and accuracy still passed 20/20;
only the performance stage failed, so the report carried an accuracy-only score
that looks healthy and is not a score. Running on the host as the invoking user
removes both: the interpreter that imports Torch is the one that compiled the
extension, and the card named is the card used.

The extension check is a *call*, not an existence test, for the same reason. A
present `_C.so` proves the build finished, not that it fits the loaded Torch.

## Configuration

One strict NAME=value file, `env.sh`, beside `env.example.sh`. Pass an absolute
path with `--env-file` (mapped to `BSKA_NPU_ENV_FILE`) or keep the git-ignored
`env.sh` in place. The Gate parses it as data and never executes it.

Required:

| Key | Meaning |
| --- | --- |
| `BSKA_NPU_CANN_SETUP` | `set_env.sh` of a toolkit complete enough for Ascend950PR: `libhccl.so` resolves and `ASCEND_OPP_PATH` carries the 950 operator package |
| `BSKA_NPU_CONDA_ENV` | the environment holding torch, torch_npu, triton, helix |
| `BSKA_NPU_CONDA_ENVS_PATH` | its parent directory |
| `BSKA_NPU_CANNBENCH_REPO` | the checkout holding `tools/run_cannbench.py` and `operators/` |
| `BSKA_NPU_CANNBENCH_TASK` | the operator the score step evaluates, as `levelN/name` |

Optional: `BSKA_NPU_CANNBENCH_ARCH` (default `ascend950`),
`BSKA_NPU_BUILD_SUBDIR` (default `build`), `BSKA_NPU_NODE_BIN`,
`BSKA_NPU_CLAUDE_BIN`, `BSKA_NPU_TRITON_ASCEND_VERSION`.

`BSKA_NPU_TRITON_ASCEND_VERSION` is the exact distribution version the Gate
enforces. Leaving it empty is refused: the Adapter has no way to prove the
triton_ascend build is the reviewed one.

`BSKA_NPU_DEVICE_ID` is deliberately not in the file. It authorizes one run and
is passed as `--device-id` or an export.

## One-time preparation

The Gate verifies the environment; it does not build it. Prepare once:

```bash
source <toolkit>/set_env.sh
PYTHON=<conda-env>/bin/python <cannbench-repo>/tools/build_cannbench_utils.sh ascend950
```

The extension lands in `<cannbench-repo>/.cannbench-build/cann_bench_utils`.
`--check-config` fails closed with the exact command when it is missing.

Rebuild after changing Python, Torch, torch_npu or CANN: the extension links
against a specific Torch, and `ldd` on `_C.so` names which one.

## Usage

```bash
python3 -B scripts/bska_gate.py --check-config
BSKA_NPU_DEVICE_ID=2 python3 -B scripts/bska_gate.py --smoke
BSKA_NPU_DEVICE_ID=2 python3 -B scripts/bska_gate.py --cannbench
```

`--check-config` is offline and fast: it reads the machine file and stats the
paths. It neither imports Torch nor touches a card.

Through AgentGates:

```bash
agent-gates doctor --adapter bska-npu --source <bska-checkout> --env-file <abs>/env.sh
agent-gates run --adapter bska-npu --source <bska-checkout> \
    --env-file <abs>/env.sh --device-id 2 --all
```

## Reading the result

Standard output is one JSON object; every diagnostic goes to stderr. Exit code
75 means the machine cannot run the Gate (BLOCKED_ENV, not a failed test).

Check `exitCodeZero` and, for the score step, `officialReportValid` — not the
score alone. A report with `score: 0.0` and `caseCount: 20` is a run whose cases
never produced timings; `timedCases` distinguishes it from a genuine pass.
