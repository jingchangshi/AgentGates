# env.sh 配置指南（Host-First 宿主机模式）

所有活跃 Adapter 读取机器配置的方式完全一致：**严格从各自的 `env.sh` 数据文件读取**，从不执行 shell 展开或命令替换（`$(...)`、`` `...` `` 等均视为字面字符串）。

| Adapter ID | 默认 `env.sh` 路径 | 覆盖选择器环境变量（或 CLI `--env-file`） |
| --- | --- | --- |
| `ascendnpu-ir`（宿主机路由） | `agent_gates/adapters/ascendnpu-ir/env.sh` | `ASCENDNPU_ENV_FILE` |
| `tilelang-npu` | `agent_gates/adapters/tilelang-npu/env.sh` | `TILELANG_NPU_ENV_FILE` |
| `tilefoundry-npu` | `agent_gates/adapters/tilefoundry-npu/env.sh` | `TILEFOUNDRY_NPU_ENV_FILE` |
| `tla-npu` | `agent_gates/adapters/tla-npu/env.sh` | `TLA_NPU_ENV_FILE` |
| `forgekernel-ascend-npu` | `agent_gates/adapters/forgekernel-ascend-npu/env.sh` | `FK_ASCEND_ENV_FILE` |
| `bska-npu` | `agent_gates/adapters/bska-npu/env.sh` | `BSKA_NPU_ENV_FILE` |

> [!WARNING]
> **Docker 模式已隔离（`DOCKER_MODE_DISABLED`）**：`tilelang-hivm`、`tilelang-hivm-npu` 及 `ascendnpu-ir` 容器路由已禁用。本文仅说明当前活跃的 6 个宿主机 Adapter 配置。

---

## 通用规则

1. **只写 `NAME=VALUE` 数据行**：支持 `#` 注释和空行；成对单/双引号会被去除，未闭合引号报错；未知键或重复键报错。
2. **禁止在外部 shell 冲突 export**：若启动 shell 中导出了同前缀变量且值与 `env.sh` 不一致，Gate 会明确拒绝，保证引擎加锁路径与 Gate 实际读写路径一致。
3. **卡号不写入 `env.sh`**：`*_DEVICE_ID` 和 `*_REQUIRE_EXCLUSIVE_DEVICE` 属于单次运行/集群策略参数，由 `--device-id` 或 `~/.config/agent-gates/config.toml` 解析后注入，绝不写入 `env.sh`。
4. **不要将 `*_BUILD_SUBDIR=` 或 `*_NINJA_JOBS=` 留成空赋值行**：若使用默认值（如 `build` 或 `$(nproc)`），应保留默认值（如 `=build`）或整行 `#` 注释掉。

---

## 一、`tilefoundry-npu`（TileFoundry）

模板文件：[`env.example.sh`](../../agent_gates/adapters/tilefoundry-npu/env.example.sh)

| 字段 | 必填 | 格式与默认值 | 作用与注意事项 |
| --- | --- | --- | --- |
| `TILEFOUNDRY_NPU_CANN_SETUP` | 是 | 宿主机绝对路径 | CANN `set_env.sh`；仅 `tilefoundry-npu-device` 会 `source` 它，`tilefoundry-npu-cpu` 刻意跳过以避免 `ortools` 与 CANN 原生库冲突 |
| `TILEFOUNDRY_NPU_CONDA_SH` | 是 | 宿主机绝对路径 | Conda `conda.sh` |
| `TILEFOUNDRY_NPU_CONDA_ENV` | 是 | 环境名 | 装有 `torch (+cpu)`、`torch_npu`、`pytest`、`z3` 的 Conda 环境名（如 `tilefoundry-npu`） |
| `TILEFOUNDRY_NPU_CONDA_ENVS_PATH` | 可选 | 宿主机绝对路径 | 环境位于自定义用户 `.conda/envs` 目录时填写 |
| `TILEFOUNDRY_NPU_PYTHON` | 可选 | 宿主机绝对路径 | 目标 Conda 环境的 `python3`；留空则由上述 Conda 路径自动拼接 |
| `TILEFOUNDRY_NPU_TILELANG_SOURCE` | 可选 | 宿主机绝对路径 | 复用 `tilelang-npu` 编译管理的宿主 `tilelang` 源码仓；填后自动前置到 `PYTHONPATH`（`tilefoundry-npu` 只读复用、绝不自编译 `tilelang`） |
| `TILEFOUNDRY_NPU_TILELANG_BUILD` | 可选 | 宿主机绝对路径 | 须等于 `<TILELANG_SOURCE>/build`（留空默认取 `<TILELANG_SOURCE>/build`）；自检与各门禁校验 `libtilelang.so`、全部 `*.abi3.so` 与 `.agent_gates_build_stamp.json`，若 `tilelang` C++/CMake/Cython 源码更新后未重编则返回 `BLOCKED_ENV` 提示先执行 `tilelang-npu-build` |
| `TILEFOUNDRY_NPU_CPU_TESTS` | 可选 | 逗号或空白分隔的**相对**路径列表 | 相对于 `--source` 的 `pytest` 测试文件或目录（禁止 `/` 开头和 `..`）。留空默认运行 5 个目录：`tests/tilefoundry,tests/ascend/target,tests/ascend/ir,tests/ascend/passes,tests/ascend/operators` |
| `TILEFOUNDRY_NPU_DEVICE_TESTS` | 可选 | 逗号或空白分隔的**相对**路径列表 | 相对于 `--source` 的额外 NPU `pytest` 路径（禁止 `/` 开头和 `..`）。**默认留空**：只跑内置窗口期 NPU 探针（`import tilefoundry`、FP32 向量加、BF16 Cube `matmul`、显存分配 > 0、`torch.npu.synchronize()`）；填写后会在同一设备窗口内追加执行 `pytest` 并要求至少通过 1 个用例且零失败 |
| `TILEFOUNDRY_NPU_BUILD_SUBDIR` | 可选 | 单级目录名（默认 `build`） | 引擎加 `build:<dir>` 锁使用；模板默认填 `build`，请勿改为 `=` 空值 |
| `TILEFOUNDRY_NPU_CANNBENCH_REPO` | CANNBench 必填 | 宿主机绝对路径 | 只读 CANNBench 仓库根路径（如 `cannbot-ops`）；门禁绝不写入该目录 |
| `TILEFOUNDRY_NPU_WORK_ROOT` | CANNBench 必填 | 宿主机绝对路径 | 须位于 `--source` 与 `CANNBENCH_REPO` 之外，存放按 Python/Torch ABI 隔离构建的 `cann_bench_utils` 与 `reports/` |
| `TILEFOUNDRY_NPU_CANNBENCH_ARCH` | CANNBench 可选 | 目标架构（默认 `ascend950`） | 可选 `ascend950`、`ascend910b`、`ascend910_93` |
| `TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE` | CANNBench 可选 | 相对 `--source` 路径 | 默认 `examples/ascend950pr_ops/cannbench_targets.json`，优先从仓内读取待测 `(task, candidate)` |
| `TILEFOUNDRY_NPU_CANNBENCH_TARGETS` | 兜底必填 | `task:rel_path` 列表 | 当仓内 targets 文件不存在时启用，如 `level4/attention_residual:examples/ascend950pr_ops/attention_residual/kernels.py` |

---

## 二、`tilelang-npu`（TileLang Ascend 宿主机版）

模板文件：[`env.example.sh`](../../agent_gates/adapters/tilelang-npu/env.example.sh)

| 字段 | 必填 | 格式与默认值 | 作用与注意事项 |
| --- | --- | --- | --- |
| `TILELANG_NPU_CANN_SETUP` | 是 | 宿主机绝对路径 | CANN `set_env.sh`（先于 Conda 加载） |
| `TILELANG_NPU_CONDA_SH` | 是 | 宿主机绝对路径 | Conda `conda.sh` |
| `TILELANG_NPU_CONDA_ENV` | 是 | 环境名 | 装有 `torch`/`torch_npu`/编译依赖的 Conda 环境 |
| `TILELANG_NPU_CONDA_ENVS_PATH` | 是 | 宿主机绝对路径 | 自定义 `.conda/envs` 根目录 |
| `TILELANG_NPU_PYTHON` | 是 | 宿主机绝对路径 | 目标 Conda 环境内的 `python3`（传入 CMake；编译产物采用 `cp39-abi3` `.abi3.so` 并写入 `<build>/.agent_gates_build_stamp.json` 供下游复用） |
| `TILELANG_NPU_CMAKE_BIN` | 可选 | 宿主机绝对路径 | 指定 `cmake`；留空使用激活环境中的 `cmake` |
| `TILELANG_NPU_BUILD_SUBDIR` | 可选 | 单级目录名（默认 `build`） | **注意**：模板文件为满足打包契约该行为空；拷贝成真实 `env.sh` 时请设为 `build` 或注释掉该行 |
| `TILELANG_NPU_NINJA_JOBS` | 可选 | 正整数 | 拷贝成真实 `env.sh` 时请按主机内存填写正整数（如 `32`）或注释掉该行，不可留空赋值 |
| `TILELANG_NPU_EXAMPLE_TESTS` | 可选 | 逗号分隔的**相对** `.py` 文件列表 | 指定 `tilelang-npu-examples` 要运行的示例；留空默认运行 3 个经审查的 Ascend 示例：`examples/ascend/test_vecadd.py,examples/ascend/test_rmsnorm.py,examples/ascend/test_per_token_cast_to_fp8.py` |

---

## 三、`tla-npu`（triton-ascend TLA 接口 E2E）

模板文件：[`env.example.sh`](../../agent_gates/adapters/tla-npu/env.example.sh)

- 宿主机直跑路径：只需填写宿主机路径；`TLA_NPU_CONTAINER_*` 系列键留空（仅保留解析兼容，Host 模式忽略）。
- 关键测试与缓存字段：
  - `TLA_NPU_CANN_SETUP`、`TLA_NPU_CONDA_SH`、`TLA_NPU_CONDA_ENV`、`TLA_NPU_CONDA_ENVS_PATH`：宿主机 CANN 与 Conda 环境（需安装含 `@tla.jit` 支持的 `triton-ascend`）。
  - `TLA_NPU_TEST_DIR`（**必填**）：宿主机绝对路径，存放外部独立 `test_tla.py` 的目录；**必须独立于 `--source` 传入的 `triton-ascend` 仓库和 `TLA_NPU_CACHE_DIR`**。
  - `TLA_NPU_CACHE_DIR`（**必填**）：宿主机绝对路径，必须位于 `--source` 与 `TLA_NPU_TEST_DIR` 之外；每次运行都在其下新建独立临时子目录作为 `TRITON_CACHE_DIR`。
  - `TLA_NPU_BUILD_SUBDIR`：不写源码内构建树，保持 `#` 注释即使用租约名 `build`。

---

## 四、`bska-npu`（cannbot-ops / CANNBench）

模板文件：[`env.example.sh`](../../agent_gates/adapters/bska-npu/env.example.sh)

- 必填键：
  - `BSKA_NPU_CANN_SETUP`：宿主机 CANN `set_env.sh`（支持 Ascend950PR）。
  - `BSKA_NPU_CONDA_ENV` & `BSKA_NPU_CONDA_ENVS_PATH`：直接调用 `<CONDA_ENVS_PATH>/<CONDA_ENV>/bin/python`。
  - `BSKA_NPU_CANNBENCH_REPO`：CANNBench 仓库绝对路径（包含 `tools/run_cannbench.py`、`operators/`、`.cannbench-build/cann_bench_utils`）。
  - `BSKA_NPU_CANNBENCH_TASK`：`bska-npu-cannbench` 门禁评测的算子任务标识，格式为 `levelN/name`（例如 `level1/swi_glu`）。
  - `BSKA_NPU_TRITON_ASCEND_VERSION`：**必填**，必须精确等于环境中 `importlib.metadata.version("triton-ascend")` 的版本字符串（如 `3.6.0.post...`），留空或不匹配会被拒绝。
- 可选键：`BSKA_NPU_CANNBENCH_ARCH`（默认 `ascend950`，可选 `ascend910b` / `ascend910_93`）、`BSKA_NPU_BUILD_SUBDIR`（保持注释或设为 `build`）、`BSKA_NPU_NODE_BIN`、`BSKA_NPU_CLAUDE_BIN`。

---

## 五、`forgekernel-ascend-npu`（ForgeKernel）

模板文件：[`env.example.sh`](../../agent_gates/adapters/forgekernel-ascend-npu/env.example.sh)

- 固定探针：无自定义测试列表字段——`smoke` 执行固定 `fk_device_probe.py`（FP32/BF16/Reduce），`build` 在隔离 venv 内安装并做版本/入口探测。
- 关键路径与策略键：
  - `FK_ASCEND_CANN_SETUP`、`FK_ASCEND_PYTHON`（宿主机基础 Python，用于创建 `--system-site-packages` venv）。
  - `FK_ASCEND_TILELANG_SOURCE` & `FK_ASCEND_TILELANG_BUILD`（**必填**，复用既有 TileLang 源码和 `build/libtilelang*.so`，只读校验哈希从不重编 TileLang）。
  - `FK_ASCEND_WORK_ROOT` & `FK_ASCEND_VENV`（**必填**，位于 `--source` 之外的运行期工作目录，保存 `build-src/<commit>/` 源码拷贝及 venv）。
  - `FK_ASCEND_DEPENDENCY_POLICY`：`frozen`（推荐）或 `online`；Agent 键 `FK_ASCEND_CLAUDE_BIN` / `_PROVIDER_*` 中 API Key 通过 `FK_ASCEND_PROVIDER_KEY_VARIABLE` 指定环境变量名，严禁把真实秘钥写入 `env.sh`。

---

## 六、`ascendnpu-ir`（宿主机路由 `ascend-build-host` / `ascend-regression-host`）

模板文件：[`env.example.sh`](../../agent_gates/adapters/ascendnpu-ir/env.example.sh)

- Ninja 目标固化在 `manifest.json`（构建：`bishengir-opt bishengir-compile install-bishengir-publish-products`；回归：`check-bishengir`）。
- 宿主机运行留空 `ASCENDNPU_CONTAINER_*`，仅配置宿主机路径：`ASCENDNPU_CANN_SETUP`、`ASCENDNPU_CONDA_SH` / `_CONDA_ENV`（若用系统 Python 可留空）、`ASCENDNPU_CCEC_BIN`（仅从零构建需要）、`ASCENDNPU_BUILD_SUBDIR`（默认 `build`）、`ASCENDNPU_NINJA_JOBS`（正整数并行度）。
- 宿主机体检还会验证有效 `PATH` 上的工具与版本：`ninja >= 1.12`、`cmake >= 3.28`、`clang`、`clang++`、`g++`、`ld.lld`、`git` 等。

---

## 七、配置自检与常见报错排查

在正式提交任务前，务必使用 `doctor` 验证对应 Adapter（不触发编译或上卡任务）：

```bash
.venv/bin/agent-gates doctor \
  --adapter <adapter-id> \
  --source /abs/path/to/repo \
  [--gate <gate-id>] \
  [--device-id <npu-id>] \
  --json
```

| 报错特征 | 原因 | 处理方式 |
| --- | --- | --- |
| `set but empty` | 在真实 `env.sh` 中写了空的 `*_BUILD_SUBDIR=` 或 `*_NINJA_JOBS=` | 填入具体值（如 `build` / `32`）或整行 `#` 注释 |
| `is exported as ..., but machine values are read only from ...` | 外部 shell `export` 与 `env.sh` 冲突 | `unset` 外部冲突变量或同步更新 `env.sh` |
| `test path must be relative inside source root` | `TILEFOUNDRY_NPU_*_TESTS` 或 `TILELANG_NPU_EXAMPLE_TESTS` 写了绝对路径或 `..` | 改为相对 `--source` 的路径 |
| `DOCKER_MODE_DISABLED` | 调用了容器 Adapter 或 `ascend-build` 等容器 Gate | 改用宿主机 Adapter 或 `ascend-build-host` / `ascend-regression-host` |
| `npu <id> is busy`（exit 75） | 开启独占模式且目标 NPU 正被其他进程占用 | 等待卡空闲或调整卡号/独占策略 |
