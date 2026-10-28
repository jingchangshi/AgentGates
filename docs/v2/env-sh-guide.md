# env.sh 配置指南（本服务器）

**三个 Adapter 读取机器配置的方式完全一致：只读各自的 `env.sh`。**

| Adapter | 机器配置文件 |
|---|---|
| `ascendnpu-ir` | `agent_gates/adapters/ascendnpu-ir/env.sh` |
| `tilelang-hivm` | `agent_gates/adapters/tilelang-hivm/env.sh` |
| `tilelang-hivm-npu` | 与 `tilelang-hivm` **共用**同一文件 |

启动 shell 里的 export **不作为取值来源**。若 export 了与文件不一致的值，Gate 会明确拒绝，
而不是悄悄二选一：

```
ASCENDNPU_BUILD_SUBDIR is build-release in the launching environment but build in
.../env.sh; machine values are read from that file only. Remove the export or edit the file.
```

这条规则不是为严格而严格：引擎按**同一份文件**解析构建目录并据此加锁。若两者各说各话，
两个并发运行可能锁住 `build` 却都在写 `build-release`。

`env.sh` 被 `.gitignore` 忽略，不提交、不随 Pack 升级覆盖。

---

## 一、ascendnpu-ir

### 本服务器可用的内容

```bash
# 容器身份（容器必须正在运行，用户/家目录真实存在）
ASCENDNPU_CONTAINER_NAME=s00653124-tilelang-hivm
ASCENDNPU_CONTAINER_USER=s00653124
ASCENDNPU_CONTAINER_HOME=/home/s00653124

# 构建目录与并行度（引擎用同一份文件解析并加锁）
ASCENDNPU_BUILD_SUBDIR=build
ASCENDNPU_NINJA_JOBS=64

# ccec 编译器 bin：仅「从零全量构建」需要
ASCENDNPU_CCEC_BIN=/home/s00653124/workspace/ccec-toolkits/default/ccec_compiler/bin

# 构建前加载 CANN，供 CMake 解析 shmem 头文件路径
ASCENDNPU_CANN_SETUP=/data/pri/cann-9.1.0/set_env.sh
# 本机全量构建和回归使用容器的系统 Python 3.12；无需 Conda
ASCENDNPU_CONDA_SH=
ASCENDNPU_CONDA_ENV=
```

也可用 `ASCENDNPU_ENV_FILE=/abs/path` 选择别处的文件——它只**选择文件**，不是参数来源。

### 增量与从零构建

`ascend-build` 按构建树状态自动选择，无需人工干预：

| 目标目录状态 | Gate 行为 |
|---|---|
| 有 `build.ninja` | 增量 `ninja`（原行为）|
| 既无 `build.ninja` 也无 `CMakeCache.txt` | **从零全量构建**：cmake 配置 + 全量编译 + 安装发布产物 |
| 有 `CMakeCache.txt` 但无 `build.ninja` | 拒绝：既不能增量，也不能重配而不删除别人创建的状态 |

从零构建调用仓库内固定的 `scripts/full-build.sh`，等价于：

```bash
bash ./build-tools/build.sh \
  --c-compiler clang --cxx-compiler clang++ \
  --enable-lld --build-type Release --enable-assertion \
  -t --bisheng-compiler "$ASCENDNPU_CCEC_BIN" \
  --disable-werror --disable-bishengir-werror \
  --build-triton --build-shmem-template \
  --build ./build -j "$ASCENDNPU_NINJA_JOBS"
```

`-t` 在源码脚本里是 `--build-bishengir-template`（模板构建），不是测试开关。
`full-build.sh` 拒绝 `-r/--rebuild`（那会清理构建目录），也不覆盖已有 `build.log`——
已存在时写入带时间戳的新文件。

`ascend-regression` **不会**触发全量构建：它只接受已存在的构建树。

### 超时链

| 层 | 值 |
|---|---|
| 全量构建容器侧 | 21600 s（6 h）|
| 构建 recipe 外层 | 23400 s |
| 增量容器侧 | 840 s（原值）|
| worker backstop | 23700 s，由已安装 manifest 推导，非硬编码 |

worker 单元会**拒绝**比最长 recipe 还小的 backstop，避免装出一个会静默截断全量构建的单元。

### 自检

```bash
.venv/bin/python -B agent_gates/adapters/ascendnpu-ir/scripts/docker_gate.py --check-config
# 期望 exit 0，末行 configuration valid (container=... user=...)
```

---

## 二、tilelang-hivm / tilelang-hivm-npu（共用一个文件）

```bash
TILELANG_HIVM_CONTAINER_NAME=s00653124-tilelang-hivm
TILELANG_HIVM_CONTAINER_USER=s00653124
TILELANG_HIVM_CONTAINER_HOME=/home/s00653124

TILELANG_HIVM_CANN_SETUP=/data/pri/cann-9.1.0/set_env.sh
TILELANG_HIVM_CONDA_SH=/data/miniconda3/etc/profile.d/conda.sh
TILELANG_HIVM_CONDA_ENV=s00653124_tilelang
TILELANG_HIVM_CONDA_ENVS_PATH=/home/s00653124/.conda/envs

TILELANG_HIVM_HIVM_COMPILER=/home/s00653124/workspace/AscendNPU-IR-1/build/bin/bishengir-compile
TILELANG_HIVM_HIVM_TARGET=Ascend950PR_9579

# 可选：构建目录与并行度
TILELANG_HIVM_BUILD_SUBDIR=build
TILELANG_HIVM_NINJA_JOBS=64

# 可选：容器内源码路径（仅当容器看到的路径与 bind-mount 推导值不同时才写）
# 它只能确认路径、不能凭空指定：挂载映射仍会推导并比对，不一致即拒绝。
# TILELANG_HIVM_CONTAINER_SOURCE=/workspace/tilelang
```

以上键全部是**文件键**，两个 TileLang Adapter 共用同一份 `env.sh`。启动 shell 里的
`export` 不是取值来源：只 export 不写文件会明确报错，值与文件冲突同样拒绝。
仍从环境读取的只有 `TILELANG_HIVM_ENV_FILE`（选文件）。

`TILELANG_HIVM_DEVICE_ID` **不写在这里**：设备由 AgentGates 本地配置解析后注入，
工具调用不能指定。

---

## 三、校验与常见错误

配置是**数据**，不是脚本：只接受 `NAME=value`、注释和空行，不执行命令替换。
未知键、重复键、不平衡引号都会明确报错。

| 报错 | 原因 | 处理 |
|---|---|---|
| `does not define ASCENDNPU_CONTAINER_NAME` | 文件缺容器身份 | 补齐三行 |
| `is exported as ..., but machine values are read only from ...` | shell 有冲突 export | `unset` 后改文件 |
| `sets unknown key` | 键名拼错 | 按本文更正 |
| `container ... is not running` | 容器未启动 | `docker start <name>` |
| `does not match the bind mount` | 源码未挂进容器 | 检查容器挂载 |

手动执行 NPU 脚本仍需自备 `TILELANG_HIVM_DEVICE_ID`；通过 AgentGates 提交时由本地配置解析。
