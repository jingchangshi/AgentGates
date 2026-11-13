# Docker 模式禁用与隔离记录

**当前策略：`AgentGates` 全仓库禁用并隔离 Docker / 容器执行模式，所有活跃项目 adapter 均使用宿主机（CANN + Conda / venv）模式运行。**

---

## 一、历史实测暴露的问题与根因

以下问题均在本机整改与端到端验证过程中实际观测并定位：

1. **容器以 `root` 身份挂载宿主机目录导致宿主机文件属主污染与 `PermissionError`**
   - 旧容器入口存在 `docker exec -u root`（如 `agent_gates/adapters/tilelang-hivm-npu/scripts/npu_gate.py`）及读写挂载宿主机 `~/.conda`、工作区的做法。
   - **实测故障 1（`tla-npu`）**：
     - 路径：`/home/s00653124/.conda/envs/s00653124_tla/lib/python3.11/site-packages/triton/backends/ascend/bin`
     - 现象：该目录及其中的 `JE2UPQAZQQEZSBSD4QQALDNCTP2MOTKGBPPG5CE3Z3V7BRTZSG7A` / `JK7SGT4HZKU6SGKZN5KH4WDOOPUZCA533WKCYDUCEEWVO7VSI7CQ` 为 `root:s00653124` 属主且权限为 `drwxr-sr-x`（缺少组写权限）。当宿主机模式运行 `tla-npu-e2e` 触发 Triton Ascend `bin_cache_manager` 编译主机版 `npu_utils.so` 与 launcher runtime 时，创建子目录 `.../ascend/bin/ORDOHY...` 直接抛出 `PermissionError: [Errno 13] Permission denied`。
   - **实测故障 2（`bska-npu` / `cannbot-ops`）**：
     - 路径：`/home/s00653124/workspace/cannbot-ops/operators/level4/mla/optimization/20261110-run01/refcache/operators/level4__mla/022d8693d3dfb58271baebbd6c48434bc43d961f09807bc1bc2e2f180e3217e9.native.pt`（权限 `-rw------- root:root`），以及 `operators/level4/attention_residual`、`__pycache__` 下大量 `root:root` 文件。
     - 现象：`AgentGates` 执行源码树哈希校验时直接因无读权限报错中止：`EXECUTION_FAILED: [Errno 13] Permission denied`。
   - **关联遗留**：`/home/s00653124/.conda/envs/bska-npu` 目录树也残留过 `root` 属主（同目录 `triton/backends/ascend/bin` 为 `root:s00653124`）。

2. **宿主机与容器工具链 / ELF 指纹混用冲突**
   - 容器与宿主机共享同一套 Conda `site-packages` 或源码内 `build/` 目录时，`triton.backends.ascend.utils._npu_elf_info`、glibc / 编译器指纹和 CMake 缓存会互相覆盖或失效，导致跨环境缓存损坏或构建产物不一致。

3. **容器创建脚本权限与设备暴露面过大**
   - `agent_gates/adapters/ascendnpu-ir/docker/create-container.sh`、`agent_gates/adapters/tilelang-hivm/docker/create-container.sh`、`agent_gates/adapters/tla-npu/docker/create-container.sh` 创建长期常驻容器，且将宿主机 Home / 工作区读写挂载并透传物理 NPU 设备节点（含 `--privileged` / `--network host` 选项），容易导致多卡片、多用户与多任务间边界不清。

4. **废弃的容器专用脚本硬编码**
   - `agent_gates/adapters/tilefoundry-npu/scripts/tilefoundry_gate.py` 与 `provision.py` 硬编码依赖 `/.dockerenv` 与 `docker exec -u root tf-ascend950pr`，无法安全复用，已被宿主机版 `tilefoundry_host_gate.py` 替代。

---

## 二、被隔离对象与多层级阻断清单

### 1. 已隔离的 Adapter 与 Route / Gate / Script
- **隔离 Adapter**：
  - `tilelang-hivm`（替代：宿主机版 `tilelang-npu`）
  - `tilelang-hivm-npu`（替代：宿主机版 `tilelang-npu`）
- **隔离 Gate / Recipe**：
  - `ascendnpu-ir` 中的 `ascend-build`（`ascendnpu-ir.incremental-build`）与 `ascend-regression`（`ascendnpu-ir.regression`）（替代：`ascend-build-host`、`ascend-regression-host`）
  - `tilelang-hivm-build`、`tilelang-hivm-regression`、`tilelang-hivm-npu-sanity`
- **禁止直接触发的脚本（均已从配置、引擎、执行器、MCP、CLI 及脚本入口阻断）**：
  - `agent_gates/adapters/ascendnpu-ir/scripts/docker_gate.py`
  - `agent_gates/adapters/tilelang-hivm/scripts/docker_gate.py`
  - `agent_gates/adapters/tilelang-hivm-npu/scripts/npu_gate.py`
  - `agent_gates/adapters/tilefoundry-npu/scripts/tilefoundry_gate.py`
  - `agent_gates/adapters/tilefoundry-npu/scripts/provision.py`
  - `agent_gates/adapters/{ascendnpu-ir,tilelang-hivm,tla-npu}/docker/create-container.sh`（允许 `--dry-run` 离线语法测试，非 `--dry-run` 直接以 `DOCKER_MODE_DISABLED` 拒绝）

### 2. 代码级多层 Fail-Closed 保护
- `agent_gates/docker_quarantine.py`：统一维护隔离清单与 `DOCKER_MODE_DISABLED` 拒绝逻辑。
- `agent_gates/config.py`：加载配置时拒绝注册被隔离 adapter 或在 `allowed_gates` 中声明被隔离 gate。
- `agent_gates/execution.py`：提交执行 `_resolve()` 时拒绝任何命中隔离清单的 gate。
- `agent_gates/engine.py`：`doctor()` 与 `run()` 均拒绝调用被隔离 recipe/script，返回 `BLOCKED_ENV`（不拉起子进程或 `docker`）。
- `agent_gates/mcp_server.py`：`list_workspaces`、`list_gates` 与未显式指定 `gates` 的 `doctor` 均过滤掉被隔离的 Docker gate，避免 agent 误选。
- `agent_gates/cli.py`：`list` 显式标记 `quarantined` 字段；`run --all` 跳过被隔离 gate。
- 上述 5 个 Docker Python 脚本的 `__main__` 入口及 3 个 `create-container.sh` 脚本：直接运行均以 `DOCKER_MODE_DISABLED` 拒绝退出（退出码 75 / 非零）。

---

## 三、未来重新启用 Docker 模式前的必要条件清单

若后续有明确需求恢复容器模式，必须先完成以下全部整改后再解除 `agent_gates/docker_quarantine.py` 中的隔离：
1. **严格非 root 执行**：容器内进程与 `docker exec` 禁止使用 `-u root`，强制绑定宿主机调用用户的 `--user uid:gid`。
2. **严禁读写挂载宿主机环境与缓存**：容器内的 Conda/venv、`site-packages`、`TRITON_CACHE_DIR`、`ASCEND_WORK_PATH` 与 `build/` 目录必须置于容器独立卷中，与宿主机环境彻底隔离，避免属主与 ELF/CMake 指纹污染。
3. **按运行最小化设备直通**：取消宽泛设备暴露与 `--privileged` 常驻容器模式，按授权设备号（`ASCEND_RT_VISIBLE_DEVICES` / 指定设备节点）做最小化隔离。
4. **移除硬编码容器名与旧入口**：废弃 `tilefoundry_gate.py`、`provision.py` 等历史硬编码实现，统一接入容器安全契约与门禁测试后再解封。
