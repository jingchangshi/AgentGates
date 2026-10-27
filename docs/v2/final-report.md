# AgentGates V2 — 最终交付报告

本文件回答目标 J4 的十二个问题，并给出三个仓库的提交与手工步骤。
所有结论都可追溯到 `docs/v2/test-report.md` 中的实测记录；无法实测的项目明确标注为
BLOCKED 或 NOT_TESTED，未以「应该可以」代替证据。

---

## 1. AgentGates V2 是否完整实现？

**是（除下文列出的两处环境阻塞）。**

| 目标章节 | 交付 | 位置 |
|---|---|---|
| B 结构化 API | `api_types.py`：WorkspaceInfo / AdapterInfo / GateInfo / ExecutionRequest / ExecutionStatus / ExecutionResult / GateResult / EvidenceVerification / VerificationRequirement / RequirementVerificationResult / APIError | `agent_gates/api_types.py` |
| B2 状态语义 | Execution 七个状态；Gate 六种判定；六族错误码 | `docs/v2/execution-state-machine.md` |
| B3 Receipt 语义 | **未改动**，原 HMAC / 重新验证 / STALE 逻辑全部保留 | `evidence.py`（零改动） |
| C 本机配置 | TOML，逐字段报错，路径规范化，符号链接拒绝 | `agent_gates/config.py` |
| C3 配置版本绑定 | `config_digest` 在 submit 时绑定，Worker 启动时校验 | `execution.py` / `worker.py` |
| C4 NPU 授权 | 无授权即拒绝，绝不挑选空闲卡 | `execution.py:_device_for` |
| D 异步 Worker | ExecutionStore + systemd 用户单元 + submit/status/wait/cancel/recover | `execution.py` / `worker.py` / `systemd_units.py` |
| E MCP Server | 10 个工具，inputSchema + outputSchema + structuredContent | `agent_gates/mcp_server.py` |
| E4 CLI 兼容 | 原五个动词不变，新增五个 | `cli.py` |

**关键设计点：**`Engine` 未被复制。Worker 直接调用 `engine.run()`，因此 Manifest、DAG、
Executor、Evaluator、flock 资源锁、进程监督器、`shell=False`、源码身份核对、
观测值断言全部沿用原有实现。既有 470 个测试零改动通过。

## 2. MCP Server 是否能够独立运行？

**是。** stdio 传输，单进程，无 HTTP 服务、无数据库、无开放端口。

```bash
/home/s00653124/workspace/AgentGates/.venv/bin/python -s -m agent_gates.mcp_server
```

已在真实 MCP stdio 客户端下验证（`tests/test_mcp_server.py`，8 个测试）：

- 10 个工具全部暴露 **inputSchema 与 outputSchema**；
- `structuredContent` 携带机器可读结果；
- 未知 workspace / 未知 execution / **未声明参数** 都返回错误，而不是看似成功的对象；
- 两个独立 server 进程读到同一份持久状态。

**安全要点：**未声明参数被显式拒绝。MCP SDK 默认会静默丢弃多余字段——那会让调用方
误以为传入的机器参数被采纳。实测：传 `shell_command` 得到
`undeclared argument(s): shell_command`。

## 3. systemd 异步 Worker 是否能跨 DSH 重启继续工作？

**是，已在真实用户级 systemd 上实测。**

| 实测项 | 结果 |
|---|---|
| `systemctl --user is-system-running` | `running` |
| submit 返回耗时 | **0.08–0.37 s**（不等待构建） |
| Gate 在独立单元中执行 | 是（`agent-gates-worker@<id>.service`） |
| 取消运行中的 Gate | `CANCELLED`，子进程树被杀，无完成标记，不产生 PASS |

Worker 属于用户级 systemd，**不属于** `dsh-web.service` 的 cgroup。DSH Web 或 MCP Server
重启后已提交的 Gate 继续运行，结果写入 journald 与 ExecutionStore。

## 4. Execution 和 Receipt 能否完整恢复？

**是。** 覆盖目标 D6 的八个场景，实测：

| 场景 | 结果 |
|---|---|
| 丢弃 manager 进程，新 manager 读取 | 状态正确 |
| 重启后幂等重复 submit | 返回原 executionId |
| Worker 被杀且无 receipt | `INTERRUPTED`，**绝不** PASS |
| Receipt 已写、status 回滚为 RUNNING | 采纳 receipt → `FINISHED` → 验证 PASS |
| 上次开机残留的 PID | 不被当作存活 |
| 终态与迟到事件 | 终态优先 |

恢复时综合检查 ExecutionStore、systemd 单元状态、Worker 进程身份（PID + boot_id +
cmdline，因为 PID 会复用）以及 receipt 是否真的通过验证。仅凭 PID 或历史状态文件
不可能伪造成功。

## 5. 两个 Compiler Preset 是否正常使用 MCP？

**是，30/30 实测通过。**

| Preset | 解析到 AgentGates 工具 | PTC SDK 绑定 |
|---|---|---|
| `ai-compiler` | **10/10** | **10/10** |
| `ai-compiler-magpie` | **10/10** | **10/10** |
| `standard` / `ptc` / `minimal` / `cordis` | 0/10 | 0/10 |

**实测修正了初始假设**：桥接是**按 Preset** 而非按 Agent。Preset 在注册时即挂载，
因此 DSH 进程在**任何 Agent 创建之前**就启动 2 个子进程（约 135 MiB），创建 Agent
**不再增加**进程。这既证明 stdio 方案足够，也说明：只要该 bundle 处于安装状态，
即使使用 Standard 模式也要承担这两个桥接进程。

## 6. PTC 是否获得结构化工具输出？

**是。** 十个工具在 PTC SDK 中全部绑定（见上表 10/10），返回值为 `structuredContent`
的 Pydantic 模型，不是需要模型解析的文本。

## 7. Team Task 强制完成门禁是否真正不可绕过工具层？

**是，并已做越权路径审计。**

- DSH 域：**136 个测试通过**，`agent-team/src` 每文件 100% 覆盖率。
- 审计结论：全部 `status: "completed"` 写入只有两行，且都受门禁约束；`team/task`
  事件只有一个提交路径；`updateTask` 全仓库只有一个调用点。
- 重放攻击：手工编辑日志伪造完成，会在投影阶段被拒绝。
- 策略与真实校验器一致性：**23/23** 检查通过，全部针对本机真实 receipt。

**设计要点（ADR-004）：**策略在 Team 事务**之外**执行，然后在事务**内**重新校验
revision、要求摘要与证据引用。这样既满足「外部验证耗时不能无限占用事务队列」
（实测单次校验 3.7 s / 10.8 s），又保证检查与提交之间不会被静默替换。

## 8. 原有三个 Adapter 是否通过回归？

**是。** 三个 Adapter 的 manifest 与脚本零改动，全部适配器测试通过。

真实编译器 Gate：

| Adapter | Gate | 结果 | 证据 |
|---|---|---|---|
| tilelang-hivm | build | **PASS** | run `13afbcbb…` |
| tilelang-hivm | regression | **PASS** | run `8866f8ff…`，14 个真实测试、0 跳过、原生库哈希与宿主一致 |
| ascendnpu-ir | build | **PASS** | run `da89fdd1…` |
| ascendnpu-ir | regression | **BLOCKED** | lit 全量实测 20–60 min，超过 1140 s 配方超时，未启动 |
| tilelang-hivm-npu | sanity | **BLOCKED** | 两张卡均被其他用户占用，且未授权设备 |

## 9. 原有 Claude Code 集成是否仍可使用？

**是，已用真实 v2 receipt 实测。**

hook `integrations/claude-code/hooks/task_completed.py` 在提供 `evidenceRoot` 时，对
run `8866f8ff…`（TileLang build + regression 双 Gate PASS）返回 **exit 0**，即放行。

一个使用注意：hook 默认读取**默认** evidence 根目录。V2 的部署把证据放在
`state_root/evidence` 下，因此需要在 receipt 块里写 `evidenceRoot`，或让
`evidence_root` 指向默认位置。这是配置问题，不是兼容性破坏——两者的 CLI 契约完全一致。

该集成通过 `agent-gates` CLI 的 `verify` / `report --json` 工作，两条路径的
行为与输出结构均未改变，相关测试全部通过。新增的异步动词与 MCP 表面是纯粹的增量。

## 10. 哪些测试缺乏真实环境或授权？

| 项目 | 状态 | 原因 |
|---|---|---|
| NPU 真机验收 | **BLOCKED** | 未授权设备；两张 Ascend950PR 均被占用 |
| AscendNPU-IR regression | **BLOCKED** | 实测耗时超配方超时，未启动 |
| 完整模型驱动 Team E2E | **NOT_TESTED** | 本机**无任何 LLM 凭据**，而 `dsh-base` 声明了 provider，缺凭据在启动时即致命 |

第三项**不是缺陷**：门禁本身（136 + 23 项检查）和真实 Team 服务上的 5 个真实事务
均已通过；缺的只是需要持续存活的模型会话的那部分。补跑方式见
`dsh-compiler-mode/docs/11-team-gate-e2e-scope.md`。

## 11. 本次各仓库的 Commit

**AgentGates** — 分支 `feat/v2-mcp-async`，基线 `f88da67`：

```
f9f51db  feat(api): versioned API contract, local TOML config, requirement verification
5b6ce35  feat(execution): durable async executions in a systemd user worker
a677369  feat(mcp): native MCP server and the asynchronous CLI verbs
c6f3dbd  test(mcp): drive the real MCP server over stdio
c5fad15  test(execution): pin the submit/observe/cancel/recover semantics
d5fbf0f  docs(v2): state machine, config schema, MCP schema, worker design, test report
```

**dsh-compiler-mode** — 分支 `feat/agentgates-mcp`，基线 `b83c179`：

```
4d8aaeb  feat(policy): AgentGates completion gate for Team tasks
0e42957  feat(preset): mount the AgentGates MCP bridge in both compiler presets
20ef000  fix(mcp): state the per-boot bridge cost precisely; install against the live profile
d7c84a2  fix(install): record AGENT_GATES_ROOT for the completion-policy row
bd0c5f5  test(mcp): refresh the tool-scope evidence and live-install baseline
21c41e7  test(mcp): refresh the tool-scope evidence after the final measurement run
40cf15b  feat(policy): resolve the policy through the bundle exports map; record E2E scope
b9b8440  test(e2e): pin the profile explicitly and record two harness findings
```

**deepseek-harness** — 分支 `feat/team-completion-policy`，基线 `b62215edfa`：

```
296181d1c2  feat(agent-team): generic Team task completion policy extension point
```

**全部未推送**，未改写任何远端公共历史。

## 12. 用户必须手工执行哪些步骤？

### 必做

1. **重启 DSH Web Host**，让 GUI 加载新的 row（本次会话运行于该服务之下，
   重启会终止当前会话，因此未由我执行）：
   ```bash
   systemctl --user restart dsh-web.service
   ```
   重启前已确认 `--dump-config` 中两个 preset 均含 `mcp-agentgates` 行、host 层含
   `agentgates-completion-policy` 行。

2. **如要在 GUI 中真正跑通门禁**，需要提供一个模型凭据（本机当前完全没有）：
   ```bash
   export DEEPSEEK_API_KEY=<key>   # 或 MAGPIE_API_KEY
   ```
   然后按第 10 节的说明补跑完整 Team E2E。

3. **`clangd`**：bundle 的 `lsp-stdio` 行需要它。本机原本没有，我安装到了
   `~/.local/bin/clangd`（LLVM 22.1.1）。若换机器需重新安装，否则 `tool-lsp` 行不会激活。

### 建议

4. **AscendNPU-IR regression 的配方超时**建议上调（当前 1140 s，实测全量 lit 需
   20–60 min），否则该 Gate 只会产出无意义的超时 FAIL。
5. **NPU 授权**：如要执行 NPU Gate，需在 `~/.config/agent-gates/config.toml` 的
   `[devices] authorized_npu_devices` 中显式写入设备号。空的列表意味着「不允许执行」。

### 已知的非本次问题

6. `scripts/verify.sh` 第 6 步 FAIL：live profile 的 `cordis.patch.yml` 相对预装备份多出
   20 行 `reasoningEfforts` 块，mtime 早于本次工作。属于既有漂移，按要求**未修改**
   操作者的模型配置，也未刷新备份。

---

## 证据可信边界（H3）

Receipt 使用本地用户级 HMAC。它用于发现意外修改，以及发现证据与执行记录不一致。
它**不是**远程签名、不是硬件证明，也**不能**抵御以同一 OS 用户身份运行的恶意代码。
文档中未作超出此范围的表述。
