# AgentGates V2 — 安装、卸载与故障恢复

## 前置条件

| 组件 | 要求 | 检查 |
|---|---|---|
| Python | ≥ 3.11 | `python3 -V` |
| user systemd | 可用 | `systemctl --user is-system-running` |
| Docker | 仅编译器 Adapter 需要 | `docker ps` |
| `clangd` | DSH bundle 的 `lsp-stdio` 行需要 | `which clangd` |
| MCP SDK | `pip install "mcp>=1.10,<3"` | 随包依赖 |

## 一、AgentGates

### 安装

```bash
cd /path/to/AgentGates
python3 -m venv .venv
.venv/bin/pip install -e .
```

### 配置

```bash
mkdir -p ~/.config/agent-gates
cp docs/v2/config.example.toml ~/.config/agent-gates/config.toml
chmod 600 ~/.config/agent-gates/config.toml
$EDITOR ~/.config/agent-gates/config.toml
```

完整字段说明见 `docs/v2/config-schema.md`。**每一个**机器参数（源码根目录、env 文件、
解释器、容器、设备号）都只在这里配置，任何工具调用都无法指定。

### 安装 systemd 单元

```python
from agent_gates import systemd_units
from agent_gates.config import load

usable, reason = systemd_units.available()
assert usable, reason

c = load()
systemd_units.install(c.worker.python)
```

`timeout_seconds` 省略时按已安装的 manifest 推导 `RuntimeMaxSec`
（`systemd_units.backstop_seconds()`，当前是「最长配方超时 + 300s」）。不要写死 7200：
ascend-build 现在可以从空目录做全量构建，72 分钟会把它腰斩；显式传一个小于推导值的数
会被拒绝并告诉你最小值。

容器身份、构建目录、并行度这些机器参数**不再**通过 `extra_env` 写进单元：
它们只来自各 adapter 的 `env.sh`（见 `docs/v2/env-sh-guide.md`）。`extra_env` 现在只
用于确实由 worker 环境决定的开关。

### 从旧单元迁移

旧单元里可能有 `Environment=ASCENDNPU_CONTAINER_*`。新版本会把它当作「文件里没有、
环境里有」的配置而拒绝启动，所以迁移顺序是：

1. 先写好 `agent_gates/adapters/ascendnpu-ir/env.sh`（含三个 `ASCENDNPU_CONTAINER_*`）。
2. 运行 `.venv/bin/python -B agent_gates/adapters/ascendnpu-ir/scripts/docker_gate.py --check-config`，
   确认输出 `configuration valid`。
3. 再重新 `install()` 覆盖单元（不带 `extra_env`），并 `systemctl --user daemon-reload`。
4. 停掉正在跑的 `agent-gates-worker@*`，否则它们仍带着旧环境。

### 自检

```bash
.venv/bin/python -m agent_gates.cli list --json
.venv/bin/python -m agent_gates.cli doctor --adapter tilelang-hivm \
    --source /path/to/tilelang --json
```

### 卸载

```bash
systemctl --user stop "agent-gates-worker@*"
rm ~/.config/systemd/user/agent-gates-worker@.service
systemctl --user daemon-reload
```

状态与证据保留在 `state_root` 与 `evidence_root` 下，删除即彻底清除。

## 二、dsh-compiler-mode

### 安装

```bash
cd /path/to/dsh-compiler-mode
scripts/install.sh            # link bundle + mcp-client + write ~/.dsh/.env
systemctl --user restart dsh-web.service
scripts/verify.sh
```

`install.sh` 会写入三个变量到 `~/.dsh/.env`：

```
AGENT_GATES_MCP_PYTHON=/path/to/AgentGates/.venv/bin/python
AGENT_GATES_CONFIG=/home/<user>/.config/agent-gates/config.toml
AGENT_GATES_ROOT=/path/to/AgentGates
```

### 验证

```bash
cd /path/to/dsh-compiler-mode
node tests/mcp/tool-scope.mjs          # 30 checks, 六个 preset 的工具可见性与资源占用
node --experimental-strip-types tests/mcp/policy.mjs   # 23 checks, 策略 vs 真实 receipt
```

### 卸载与回滚

```bash
scripts/uninstall.sh           # 取消选中 + 移除链接 + 清理全部 AGENT_GATES_* 行
scripts/rollback.sh            # 从预装备份恢复 profile
systemctl --user restart dsh-web.service
```

`uninstall.sh` 删除**所有** `AGENT_GATES_*` 行而不是枚举的那几个，这样日后新增的变量
也不会残留。

## 三、故障诊断

### Gate 停在 SUBMITTED / STARTING

```bash
journalctl --user -u "agent-gates-worker@<executionId>.service" -n 50
```

| journal 关键字 | 含义 | 处理 |
|---|---|---|
| `configuration unusable` | 配置文件缺失或非法 | 检查 `AGENT_GATES_CONFIG` |
| `ModuleNotFoundError: No module named agent_gates` | 解释器找不到包 | 检查单元的 `PYTHONPATH` |
| `CONFIG_IDENTITY_CHANGED` | 提交后配置被改 | 重新提交（这是设计行为） |

### Execution 变成 INTERRUPTED

表示「无法证明结果」，不是「失败」。常见原因：Worker 被 `SIGKILL`、机器重启、
Worker 退出但未发布 receipt。查看 `state_root/executions/<id>/` 下的记录；若 receipt
实际已存在，下一次 `status()` 会自动采纳它。

### 校验失败 `STALE`

receipt 生成后源码、adapter、配置或构建产物发生了变化。这是**正确行为**：历史 PASS
不能用于当前源码。重新提交 Gate 即可。

### `RESOURCE_BUSY`

另一个 AgentGates 运行持有同一源码或构建目录的 `flock`。等待其结束后重新提交。

### MCP 工具在 GUI 中看不到

1. `node apps/cli/lib/bin.js --profile web --dump-config | grep mcp-agentgates`
2. 确认 `~/.dsh/.env` 中三个 `AGENT_GATES_*` 变量存在且为绝对路径
3. 重启 `systemctl --user restart dsh-web.service`

## 四、已知限制

- **HMAC 的可信边界**：检测意外修改与不一致，不抵御同用户恶意代码，不是远程签名。
- **桥接进程按 Preset**：只要 bundle 安装，任何会话（含 Standard）都承担 2 个 MCP 子进程。
- **完成门禁只校验提交时的源码**：任务完成后若源码继续变化，receipt 会转 STALE，
  需在最终交付前重新验证。
- **NPU 必须显式授权**：空闲卡不等于授权卡。
