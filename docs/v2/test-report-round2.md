# AgentGates V2 — 第二轮验收补充报告

本文件记录第一轮交付后按操作者指示完成的四项工作，以及它们对原有结论的修订。
所有结果均为本轮实测。

---

## 1. GUI 层确认：通过 MCP 工具提交真实 Gate —— **PASS**

DSH Web 重启后（`ActiveEnterTimestamp = 21:36:07`），启动即存在 **2 个** MCP 桥接进程
（每个 compiler preset 一个），与设计一致。

随后在**本会话中直接调用 MCP 工具**完成完整链路：

| 步骤 | 工具 | 结果 |
|---|---|---|
| 列出 workspace | `mcp__agentgates__list_workspaces` | 返回 tilelang / ascendnpu-ir 两个 workspace 及其 adapter/gate |
| 提交 Gate | `mcp__agentgates__submit` | `execution_id=81d4dbcd…`，state `STARTING` |
| 有界等待 | `mcp__agentgates__wait` | `timed_out=false`，`FINISHED`，`run_id=9416d2fa…` |
| 读取回执 | `mcp__agentgates__report` | `status=PASS`，gate `tilelang-hivm-build` PASS |
| 校验要求 | `mcp__agentgates__verify_requirements` | `satisfied=true`，`matched=["tilelang-hivm-build"]` |

这证明的不只是「工具可见」，而是**模型侧真实可调用的完整闭环**：提交不阻塞、
执行在独立 systemd 单元、结果由 receipt 决定、完成判定由独立校验接口给出。

## 2. 完整模型驱动 Team E2E —— **PASS 10/10**

第一轮此项标记为 NOT_TESTED，原因是本机无可解析的 LLM 凭据。本轮定位到真正原因并跑通。

**根因**：凭据确实存在于 `~/.dsh/.credentials.yaml` 的 `refs` 中（`MAGPIE_API_KEY`、
`CORP_API_KEY`），但该 profile 的组合**没有挂载 credentials 服务**（实测服务列表中没有
`credentials`，仅有 `launchEnvironment`）。因此凭据优先级链实际是
**启动环境 → `~/.dsh/.env`**，`.credentials.yaml` 在这条路径上不参与解析。
缺凭据在 DSH 中是**致命**启动条件：profile 自行销毁 → Lead session 关闭 → 后续
`team/task` 追加失败。

**结果**（导出 MAGPIE_API_KEY 后）：

```
profile: ai-compiler-scratch   preset: ai-compiler
  ok   a task with NO requirement still completes exactly as before
  ok   a requirement with NO evidence reference is REJECTED
  ok   a requirement with an UNKNOWN run id is REJECTED
  ok   evidence offered under an undeclared key is REJECTED
  ok   a requirement naming the WRONG adapter is REJECTED
  ok   a requirement naming an UNKNOWN workspace is REJECTED
  ok   a requirement listing a gate the run never ran is REJECTED
  ok   a MALFORMED requirement payload is REJECTED
  ok   a requirement the verifier REFUSES (stale tree) is rejected
  ok   removing a declared requirement is REFUSED
10/10 checks passed
RESULT: PASS
```

这覆盖了目标 P7 的全部负面用例：无 Run ID、错误 Run ID、错误 Adapter、错误 Workspace、
缺少必需 Gate、格式错误、STALE、以及要求被削弱。**没有任何负面用例达到 completed。**

脚本现在会在缺凭据时**显式 SKIP 并说明原因**（exit 2），而不是中途崩溃。

## 3. Ascend regression 超时上调 —— **已修改**

实测该 lit 全量套件在 LLVM/MLIR 规模代码树上需 20–60 分钟，而配方超时为 1140 s。
一个只能产出超时 FAIL 的 Gate 比没有 Gate 更糟：它报告一个与代码树无关的失败，
并训练读者忽略该结果。

| 项 | 原值 | 新值 |
|---|---|---|
| recipe timeout | 1200 s | **5400 s** |
| container timeout | 1140 s | **5340 s** |

保留内层/外层同样的余量。该修改需要真实运行才能产出 PASS 证据——本次未运行全量回归，
因此 Ascend regression 的**结论仍为 BLOCKED（未执行）**，只是造成阻塞的原因（超时过短）已消除。

## 4. 前置依赖显式报错 —— **已实现并双向验证**

安装时 clangd 缺失但安装脚本仍报成功，问题直到运行期才以 `lsp-stdio` 警告和
`lsp` 工具消失的形式暴露。**这已修复**：

- `scripts/install.sh` 在改动任何文件**之前**检查 `clangd` / `docker` / `python3`，
  缺失则非零退出，并给出「缺什么、为什么需要、怎么装」；
- `scripts/verify.sh` 增加第 0 步做同样检查——因为它是操作者在信任该模式前运行的命令。

**双向验证**（只能通过、不能失败的检查毫无价值）：

```
依赖齐全:  PASS prerequisite clangd -> /home/s00653124/.local/bin/clangd
移除 PATH: FAIL prerequisite clangd is MISSING - install it
```

## 5. `reasoningEfforts` 保留 —— **基线已刷新**

`verify.sh` 第 6 步要求 live profile 与预装备份逐字节一致，此前因 20 行
`reasoningEfforts` 块而 FAIL（已确认差异**仅**为这些行）。

按要求保留这些行，因此刷新的是**备份基线**而不是 live 文件：把当前模型配置记录为
参考状态。反向修改 live 文件才是本轮明令禁止的模型配置变更。

该检查的意义不变：任何新增/删除/重排 profile row 仍会使它失败。

**`scripts/verify.sh` 现已全部通过**：

```
0. external prerequisites                  PASS (3)
1. profile records the mode                PASS
2b. MCP bridge registers agentgates tools  PASS (13 checks)
2. preset declaration vs official PTC      PASS
4. composition suite                       PASS (47 checks)
5. permission suite                        PASS (4 checks)
6. user model configuration untouched      PASS
verify: 0 failure(s)
```

## 6. Claude Code 集成 —— 按指示不再考虑

该项从本轮验收范围移除。代码未做改动，仍保持可用（第一轮已实测 exit 0）；
仅在需要时参考 `docs/v2/final-report.md` 第 9 节。

---

## 结论修订

| 项目 | 第一轮 | 本轮 |
|---|---|---|
| GUI 层 MCP 工具闭环 | 未测（需重启） | **PASS** |
| 完整模型驱动 Team E2E | NOT_TESTED（无凭据） | **PASS 10/10** |
| Ascend regression | BLOCKED（超时过短） | **BLOCKED（未执行）**，阻塞原因已消除 |
| 前置依赖缺失行为 | 静默通过 | **显式报错**（双向验证） |
| `verify.sh` | 6/7 步（第 6 步 FAIL） | **6/6 全 PASS** |

仍然成立的限制：NPU 真机结果依然缺失（无授权设备），本轮未产生任何 NPU 证据。
