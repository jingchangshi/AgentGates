# P0-E/F/G/H 可执行实现计划（task-52）

状态：**计划 + 失败基线**。本文件不改动 `scripts/tilefoundry_gate.py`（现由 I2 独占，task-50/51）。
所有"要改的位置"以**函数名 + 当前内容锚点**记录，因为 gate.py 正在被 I2 改动，行号会漂移；
等 I2 交还后按锚点重新定位再出 diff。

审计依据：`artifacts/run02/review/architect_audit_agentgates_e2e_hardening.md` §5。

---

## 0. 现状实测（不是猜测）

在 `agent_gates/adapters/tilefoundry-npu/scripts/tilefoundry_gate.py`（1397 行，I2 编辑中）上逐项 grep：

| 机制 | 现状 | 证据 |
|---|---|---|
| 团队锁 / lease / 父子交接 | **完全不存在** | `grep -n 'flock|LOCK_|lease'` → 无命中 |
| 阶段 watchdog | **不存在** | `grep -n 'watchdog|phase'` → 无命中（仅注释里的 "phase" 一词） |
| finally 收尾 after-SMI | **不存在** | `grep -n 'finally'` → 无命中 |
| 唯一 run dir | **不存在** | 只有 `run_id = uuid4().hex[:12]`，无 per-run 目录 |
| 失败也写单 JSON | **部分**：只有 `evidence_dir is not None` 且跑到末尾才写 | `write_evidence` 仅在成功路径被调用 |
| phase DAG / 完成数 | **不存在** | 无 |
| FINISHED≠PASS | **不成立**：GateError 直接抛出，无 JSON | 见 `main` 的 except |
| native/四 case recipe | **完全不存在** | device payload 只有 generic `ones+2` 加法 |
| SMI 前后各一次 | **只有 BEFORE** | `device_occupancy` 仅在第 1183 行调用一次 |

busy 语义已经是正确的（`occupied=None` 表示 UNKNOWN idle、只记录不改判定），这部分**不动**。

---

## 1. P0-E 团队串行 / 完成边界

**要改（等 I2 交还）**：
1. 新增模块级 `DEVICE_LOCK_PATH = "/home/.tf_device0.lock"`（与 TF 侧 `run_case.py` 同一把锁，
   跨仓合同，不能新造第二把）。
2. 新增 `def device_lock_lease():` — contextmanager：
   - 读 `TF_DEVICE_LOCK_HELD`：为 `"1"` 时**只继承不重入**（父子交接），返回 lease 对象，
     `handoff_from_parent=True`；
   - 否则 `fcntl.flock(LOCK_EX|LOCK_NB)`，被占时 **不等待、不重试**，抛
     `GateError("LOCK_HELD_BY_OTHER: ... ", EXIT_BLOCKED)`；
   - lease 记录 `path / pid / mode / handoff` 进证据。
3. `gate_device` 在**任何 container_exec 之前**进入 lease，`finally` 释放。
4. **禁止**外层再包 flock：lease 是唯一入口；父子场景靠 env 交接（TF 侧 `run_case.py` 已实现同一约定）。
5. 只协调**本团队**作业；busy 的他人 PID 只记录。

**要新增的测试/负例**（`tests/test_e2e_chain.py`）：
- `test_two_workers_race`：两个进程真实争锁，**恰好一个**进入 payload，另一个 `EXIT_BLOCKED`；
  断言失败者是 `LOCK_HELD_BY_OTHER` 且**没有**任何 container_exec 调用。
- `test_fake_lock_held_env`：`TF_DEVICE_LOCK_HELD=1` 但没有父进程真持锁 ⇒ 必须能识别
  （假锁冒证负例：不能仅凭 env 就认为有父作业）。
- `test_parent_child_handoff`：父持锁 + 子继承 ⇒ 子不重入、不死锁、两者都完成。
- `test_lease_released_on_failure`：payload 抛错后锁必须已释放（下一个作业能进）。

## 2. P0-E 阶段 watchdog / hang 证据

**要改**：
1. 新增 `STAGE_TIMEOUTS = {"preflight": 90, "deploy": 120, "occupancy_before": 90,
   "payload": 600, "occupancy_after": 90}`（单次有界，**不重试**）。
2. 新增 `def run_stage(name, fn, phases):` — 计时 + 超时 + 记录 phase：
   `{name, started, finished, duration_s, status, code}`。
3. 超时 ⇒ `status="TIMEOUT"` 且写 phase，**不是** PASS、不是 busy。

**测试/负例**：
- `test_stage_timeout_is_not_pass`：payload 卡住 ⇒ phases 有 `TIMEOUT`，退出码非 0，JSON 仍在。
- `test_no_retry_on_timeout`：断言 payload 只被调用 **1 次**（无重试至空闲）。

## 3. P0-E finally 一次 after-SMI

**要改**：`gate_device` 外层 `try/finally`，finally 里**恰好一次** `device_occupancy(...)`
（before 已是现有那次），写入 `phases.json` 的 `occupancy_after`；after 自身失败
⇒ 记录 `AFTER_SMI_FAILED` 但**不覆盖**已经判定出来的 verdict，也不把 FAIL 变 PASS。

**测试/负例**：
- `test_after_smi_runs_exactly_once`：计数 == 1。
- `test_after_smi_failure_keeps_verdict`：after SMI 抛错，payload 已 FAIL ⇒ 仍 FAIL（不被改写成 pass）。
- `test_after_smi_failure_on_pass`：payload PASS 但 after SMI 失败 ⇒ 结论必须是"**证据不完整**"，
  不得静默 PASS。

## 4. P0-G 完整结果 / 依赖证据

**要改**：
1. 新增 `def run_directory(evidence_dir, gate_name, run_id):` → `<evidence>/runs/<gate>-<run_id>/`，
   唯一、不复用；所有 phase 产物写进去。
2. 改成**任何**结局（PASS/FAIL/BLOCKED_EXIT/TIMEOUT/异常）都在 `finally` 写**单个** `result.json`，
   含：`run_id, gate, status, exit_code, started, finished, phases[], phase_dag, artifacts{name:sha256}`。
3. `phase_dag`：显式前置边，如 `payload <- {preflight, deploy, occupancy_before}`；
   被跳过/未跑的 phase 记 `status="SKIPPED"`，**不计入完成数**。
4. `status="FINISHED"` **不是** PASS：`result.json` 里 `passed` 由
   `all(phase.status in {PASS, RECORDED})` 且 payload 有真实观测共同决定。
5. manifest 的 `gates[].observationChecks` 增加前置声明（`requiresPhases`），
   使"跳过阶段"可被静态检出。

**测试/负例**：
- `test_failure_also_writes_one_json`
- `test_unique_run_dir_per_run`（两次运行目录不同）
- `test_finished_is_not_pass`：所有 phase 完成但 payload 无观测 ⇒ `passed=false`
- `test_skipped_phase_not_counted`
- `test_exit_zero_with_empty_artifacts_is_fail`
- `test_phase_dag_precondition_enforced`：缺前置 phase ⇒ 拒绝

## 5. P0-F 可复用 native / 四 case recipe

**要改**：
1. manifest 新增 recipe：`tilefoundry-npu.native`（vcvt/vmula/vcadd）与
   `tilefoundry-npu.cases`（decode_dyn/decode_recompute/prefill_dyn/prefill_recompute），
   与 device **同一** context/env/lock，但 compile / runtime / numeric **各自独立门**。
2. `gate_device` 不再内嵌 generic `ones+2` payload；改为 `--workload {smoke,native,cases}`。
3. 缺 case ⇒ **具名拒绝** (`CASE_MISSING`)，**绝不**回退 generic add。
4. `plan`/`archive` 读回**不得**标成 device：沿用 TF 侧 `readback_source` 字段语义。

**测试/负例**：
- `test_missing_case_refuses_not_fallback`（核心负例）
- `test_plan_readback_not_device`
- `test_compile_runtime_numeric_are_separate_gates`

## 6. P0-H 强回归

- 环境 shell / loader / device 契约测试，**真实 shell bootstrap** + stub SMI/loader/torch，
  不是把 `device_occupancy` 整个替换掉。
- 每条：正常路径先 PASS，再单缺陷，再 **probe-can-fail**（注入 bug 必须被抓到）。
- 现有静态/两状态 mock 全绿**不得**替代整链。

---

## 7. 等 I2 交还后我要动的具体位置

**范围声明（Lead 2026-11-05 职责修正后）**：本表共 **8 处**，全部属 P0-E/F/G/H。
attestation 通道（`--deployment-attest` / `--deployment-attestation` /
`verify_deployment_attestation`）的**实现归 I2（P0-DEPLOY）**，**不在本表**；
我只负责其测试侧（§9）与交还后的复评。我曾把 attestation 列为第 9–11 处，
那是越界，已删。

| # | 锚点（函数/内容，非行号） | 动作 |
|---|---|---|
| 1 | 模块顶部 `EXIT_ASSERT = 97` 之后 | 新增 `DEVICE_LOCK_PATH`、`STAGE_TIMEOUTS` |
| 2 | `def log(...)` 之后 | 新增 `device_lock_lease`、`run_directory`、`run_stage`、`write_result` |
| 3 | `gate_device` 首行 `assert_not_host` 之前 | 进入 lease（最外层） |
| 4 | `gate_device` 内 `occupancy, occupancy_raw = device_occupancy(...)` 一次调用 | 包进 `run_stage("occupancy_before")` |
| 5 | `gate_device` 内 `payload = """...ones+2..."""` 整块 | 抽出为 workload 分派，新增 native/cases |
| 6 | `gate_device` 尾部 `if evidence_dir is not None: write_evidence(...)` | 改为无条件 `write_result`（含失败路径） |
| 7 | `write_evidence` | 保留；新增 `write_result` 作为单 JSON 汇总 |
| 8 | `main` 的 `try/except GateError` | `finally` 补 after-SMI + 单 JSON + 唯一 run dir |

**不改**：`parse_npu_smi`、`device_occupancy` 的 UNKNOWN idle 语义、`parse_device_list`、
`environment_preflight` / `deployment_verify` 的判定（busy 语义已正确）。

---

## 8. 用户硬判据（贯穿全部改动）

- busy **只记录**：不等待 idle、不等他人退出、不拒绝/skip/降级、不按忙闲择卡。
- 团队锁**只协调本团队**作业，不排斥其他租户。
- 任何"等一会再试"的实现都视为缺陷；`tests/device_policy.py` 的
  `audit_source`（禁 sleep/waitpid/poll/wait + 禁 busy 分支改判定）继续作为硬门。
---

## 9. 部署边界：host-only attestation 通道（**实现归 I2，我只负责测试侧**）

`--deployment-attest` 是**只读 host 通道**：不属任一 gate、不跑设备、只做 `docker inspect`。
容器内 device gate 通过 `--deployment-attestation <file>` 消费该记录，**七项全过**才允许
`deploymentVerified=true`。

### 职责边界（Lead 2026-11-05 修正）
| 项 | 归属 |
|---|---|
| `--deployment-attest` / `--deployment-attestation` 参数 | **I2**（P0-DEPLOY） |
| `verify_deployment_attestation` 实现 | **I2** |
| 失败码 STALE/CONTAINER_MISMATCH/DEVICE_MISSING/NODE_NOT_CHAR/MOUNT_MISSING/IMAGE_DRIFT/TAMPERED | **I2** 实现 |
| 七负例 + 正面控制 + 区分力控制（测试） | **我（本文件 §9 后半）** |
| 交还后复跑并判定"7 条是否转绿" | **我** |

**我的动手清单是 8 处（P0-E/F/G/H），不含 attestation 的 3 处。** 见 §7。

### 验证器合同（测试已按此写死，供 I2 对照）
`verify_deployment_attestation(attestation, config, expected, run_start) -> dict`
- 通过返回 `{"verified": True, "refused": ""}`
- 拒绝返回 `{"verified": False, "refused": "<CODE,...>"}` 或抛 `GateError`

七项与拒绝码：
| 缺陷 | 拒绝码 |
|---|---|
| 记录过期（早于本次 run 一个窗口以上） | `ATTESTATION_STALE` |
| container_id 与当前容器不符 | `ATTESTATION_CONTAINER_MISMATCH` |
| 请求的卡未绑定 | `ATTESTATION_DEVICE_MISSING` |
| davinci 路径不是字符设备（伪字符节点） | `ATTESTATION_NODE_NOT_CHAR` |
| 缺必需 mount | `ATTESTATION_MOUNT_MISSING` |
| 镜像与合同不符 | `ATTESTATION_IMAGE_DRIFT` |
| 记录被改动（payload hash 不自洽） | `ATTESTATION_TAMPERED` |
另加 `ATTESTATION_SCHEMA` / `ATTESTATION_MALFORMED` 两个结构性拒绝。

记录字段（测试用的最小集）：`schema, attested_at, container_name, container_id,
image_id, bound_devices, nodes, node_kinds, mounts, nonce, payload_sha256`。

### 我交付的 8 条测试（已在 `test_e2e_chain.py` 落地）
- 正面控制：`test_attestation_accepts_a_well_formed_fresh_record`
- 七个负例：stale / container_id_mismatch / missing_device / pseudo_character_node /
  missing_mount / image_drift / tampered_hash
- **区分力控制（常设，不得删）**：
  `test_reference_verifier_discriminates_every_negative` —— 把七例回放到一个合同参考
  实现上，要求**各自以自己的拒绝码被拒**、且干净记录被接受。没有这条，"七项全拒"会
  被一个"什么都拒"的验证器白满足。

### I2 交还后的验收动作（我做）
1. 复跑本文件；**七负例应转绿**（正面控制与区分力控制本来已绿）。
2. 若某条不转绿：按名上报"哪条、期望拒绝码、实际结果"，视为实现缺陷，**不由我改实现**。
3. 若 I2 用了不同的失败码命名：以**行为**为准（该缺陷被拒即可），但要在报告里
   写明码名差异，避免后续把"码不同"误判成"功能缺失"。

### 与 P0-E 的关系
两者同层（都是"证明环境与本次运行对应"），因此测试放在同一文件、共用同一个真实进程
竞争模型；但锁属于团队串行，attestation 属于部署来源，二者的失败码互不复用。
