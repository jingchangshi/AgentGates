# CLI reference

推荐使用独立虚拟环境安装：`python3.11 -m pip install -e .`。
所有子命令均支持 `--json`（输出结构化 JSON）与 `--evidence-root PATH`（覆盖默认证据目录 `~/.local/state/agent-gates` 或 `config.toml` 配置）。

退出码约定：
- `0`：`PASS` / 查询成功 / `verify-requirements` 满足（`satisfied: true`）。
- `1`：已记录的非 `PASS` 结果（`FAIL` / `BLOCKED_ENV` / `STALE` / `INVALID_EVIDENCE` 或 `satisfied: false`）。
- `2`：调用参数错误、环境配置无效（`CONFIG_INVALID`）或命中已隔离的 Docker 模式（`DOCKER_MODE_DISABLED`）。

---

## 一、子命令总览

### 1. 同步检索与本地直跑命令（V1）

| 命令 | 用途 |
| --- | --- |
| `agent-gates list` | 列出已安装 Adapter、manifest SHA-256、Gate 列表及隔离状态（`quarantined` / `quarantinedGates`） |
| `agent-gates doctor --adapter ID --source ROOT [--gate GATE] [--device-id ID] [--env-file PATH]` | 在不执行编译或正式测试的前提下诊断机器 `env.sh`、工具链、CANN/Conda 与 NPU 可用性 |
| `agent-gates run --adapter ID --source ROOT (--gate GATE ... \| --all) [--device-id ID] [--env-file PATH]` | 在当前进程同步运行指定 Gate（或 Adapter 全部活跃宿主机 Gate）并落盘收据 |
| `agent-gates report [--run-id ID]` | 读取收据并根据当前源码与环境重验，返回最新状态（默认读取最近一次运行） |
| `agent-gates verify [--run-id ID]` | 重验既有收据的 HMAC 签名、manifest 摘要、当前源码树与产物哈希 |

> [!NOTE]
> 对带有隔离容器路由的 Adapter（如 `ascendnpu-ir`），`run --all` 会自动跳过受隔离的容器 Gate（`ascend-build`、`ascend-regression`），仅执行宿主机 Gate（`ascend-build-host`、`ascend-regression-host`）。显式传入 `--gate ascend-build` 等被隔离路由会立即以 exit 2（`DOCKER_MODE_DISABLED`）拒绝。

### 2. 异步控制面与验收命令（V2，推荐 Agent 使用）

V2 命令自动读取 `~/.config/agent-gates/config.toml`（或由环境变量指定的配置），无需手动传入 `--source`、`--env-file` 或 `--device-id`。

| 命令 | 用途 |
| --- | --- |
| `agent-gates submit --workspace WS --adapter ID --gate GATE [--gate ...] [--request-id KEY]` | 校验证明与并发限制后提交给 `agent-gates-worker@<execution_id>.service` 异步执行 |
| `agent-gates status --execution-id ID` | 查询单次异步执行的实时状态（`QUEUED` / `RUNNING` / `FINISHED` / `CANCELLED` / `INTERRUPTED`）及关联 `run_id` |
| `agent-gates wait --execution-id ID [--timeout SEC]` | 阻塞等待执行结束（默认 `--timeout 15.0`；超时到达未结束仍正常返回当前状态供轮询） |
| `agent-gates cancel --execution-id ID` | 取消执行并终止对应的 `systemd` worker 进程树 |
| `agent-gates verify-requirements --workspace WS --adapter ID --gate GATE [--gate ...] --run-id RUN_ID` | 基于指定 `run_id` 判定当前工作区源码状态是否**仍然满足**所有要求 Gate 的通过标准 |

---

## 二、参数与环境变量规则

1. **源码根目录（`--source`）**：
   - 必须是目标项目的真实 Git checkout（或 worktree）根目录；传入仓库子目录会被拒绝。
   - 每个 worktree 必须有自己独立的构建目录与配置，工具不会跨 worktree 借用其它目录的构建产物。
2. **机器配置文件映射（`--env-file`）**：
   - `--env-file /abs/path/env.sh` 按 Adapter 映射为对应的文件选择器变量：
     - `ascendnpu-ir` -> `ASCENDNPU_ENV_FILE`
     - `tilelang-npu` -> `TILELANG_NPU_ENV_FILE`
     - `tilefoundry-npu` -> `TILEFOUNDRY_NPU_ENV_FILE`
     - `tla-npu` -> `TLA_NPU_ENV_FILE`
     - `forgekernel-ascend-npu` -> `FK_ASCEND_ENV_FILE`
     - `bska-npu` -> `BSKA_NPU_ENV_FILE`
   - 省略 `--env-file` 时，默认查找各 Adapter 同目录下的 `agent_gates/adapters/<dir>/env.sh`。不会主动搜索 `$HOME`。
3. **设备授权（`--device-id`）**：
   - NPU 上卡门禁必须显式授权卡号：V2 模式从 `config.toml` 的 `workspaces.<ws>.npu_device` 或 `devices.default_npu_device` 解析；V1 命令行模式需传 `--device-id ID`（或与配置文件无冲突的 `<PREFIX>_DEVICE_ID`）。
   - AgentGates 不负责替多租户机器分配硬件；若配置了 `require_exclusive = true`（如 `tla-npu`、`tilefoundry-npu`、`tilelang-npu` 等读取该策略时），当 `npu-smi info` 发现目标卡上有其他占用进程或驱动不可读时，返回 `BLOCKED_ENV`（exit 75）。

---

## 三、常用命令示例

### 1. 环境自检（`doctor`）

```bash
# 检查 AscendNPU-IR 宿主机编译与回归环境
agent-gates doctor \
  --adapter ascendnpu-ir \
  --source /abs/path/to/AscendNPU-IR \
  --gate ascend-build-host \
  --gate ascend-regression-host \
  --json

# 检查 TileFoundry CPU 与 NPU 环境
agent-gates doctor \
  --adapter tilefoundry-npu \
  --source /abs/path/to/TileFoundry \
  --device-id 1 \
  --json
```

### 2. V2 异步提交、等待与严格验收

```bash
# 1) 提交 tla-npu 上卡 E2E 门禁
agent-gates submit --workspace triton-ascend --adapter tla-npu --gate tla-npu-e2e --json

# 2) 等待完成（获取 state 与 run_id）
agent-gates wait --execution-id <EXECUTION_ID> --timeout 1300 --json

# 3) 查看详细门禁观测报告（注意指向 config.toml 中的 v2 evidence_root）
agent-gates report --run-id <RUN_ID> --evidence-root ~/.local/state/agent-gates/v2/evidence --json

# 4) 终态验收：确认在当前源码下是否仍全部 PASS
agent-gates verify-requirements \
  --workspace triton-ascend \
  --adapter tla-npu \
  --gate tla-npu-e2e \
  --run-id <RUN_ID> \
  --json
```

### 3. 安装 User Systemd Worker 单元 & 启动 MCP Server

```bash
# 安装/刷新 ~/.config/systemd/user/agent-gates-worker@.service 并 daemon-reload
.venv/bin/python -m agent_gates.systemd_units --python "$PWD/.venv/bin/python"

# 启动 stdio MCP server（通常在 Agent 宿主配置中登记此命令）
.venv/bin/python -s -m agent_gates.mcp_server
```

AgentGates 的所有命令均不会擅自安装 CANN/Conda、重建镜像、清理已有编译目录或复用过时的 `PASS` 收据。
