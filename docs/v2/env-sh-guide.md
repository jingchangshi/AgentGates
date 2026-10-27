# env.sh 配置指南（本服务器）

两个 Adapter 读取机器环境的方式**不同**，这是配置时最容易踩坑的地方：

| Adapter | 机器参数来源 | 为什么 |
|---|---|---|
| `tilelang-hivm` / `tilelang-hivm-npu` | **只读 `env.sh` 文件** | 文件是唯一来源，启动 shell 里的 export 被忽略 |
| `ascendnpu-ir` | **只读进程环境变量** | 不读任何文件；值由 systemd worker 单元提供 |

这一差异是刻意的：容器身份与工具链路径属于「这台机器的事实」，应当有一个
明确、可审计、不会被某个 shell 的历史污染的位置。

---

## 一、tilelang-hivm / tilelang-hivm-npu：`env.sh`

### 位置

```
agent_gates/adapters/tilelang-hivm/env.sh
```

两个 Adapter 共用这一个文件（都读 `TILELANG_HIVM_` 前缀）。该文件被 `.gitignore`
忽略，因此不会提交、不会随 pack 升级被覆盖。

也可以放在别处，用 `TILELANG_HIVM_ENV_FILE=/abs/path/env.sh` 指定——但那个变量
必须来自单元环境，而单元是安装时写定的。

### 本服务器已验证可用的内容

```bash
# 容器身份：必须是一个**正在运行**的容器，且用户/家目录真实存在
TILELANG_HIVM_CONTAINER_NAME=s00653124-tilelang-hivm
TILELANG_HIVM_CONTAINER_USER=s00653124
TILELANG_HIVM_CONTAINER_HOME=/home/s00653124

# CANN：pytest 导入 torch_npu 需要它提供的运行库
TILELANG_HIVM_CANN_SETUP=/data/pri/cann-9.1.0/set_env.sh

# Conda：TileLang 构建所用的解释器环境
TILELANG_HIVM_CONDA_SH=/data/miniconda3/etc/profile.d/conda.sh
TILELANG_HIVM_CONDA_ENV=s00653124_tilelang
TILELANG_HIVM_CONDA_ENVS_PATH=/home/s00653124/.conda/envs

# Late-hivm 后端：回归实际调用的编译器，以及它要发射的目标
TILELANG_HIVM_HIVM_COMPILER=/home/s00653124/workspace/AscendNPU-IR-1/build/bin/bishengir-compile
TILELANG_HIVM_HIVM_TARGET=Ascend950PR_9579
```

权限建议 `chmod 600`。

### 每一项怎么确认

```bash
# 容器在跑、镜像与挂载正确
docker ps --filter name=s00653124-tilelang-hivm

# 容器内确实有 ninja / cmake / 目标 python 环境
docker exec s00653124-tilelang-hivm bash -lc \
  "which ninja cmake; conda env list | grep s00653124_tilelang"

# CANN 与 conda 路径存在
ls /data/pri/cann-9.1.0/set_env.sh /data/miniconda3/etc/profile.d/conda.sh

# HIVM 编译器存在且可执行（这是自己构建的，不是预装）
ls -l /home/s00653124/workspace/AscendNPU-IR-1/build/bin/bishengir-compile
```

### 关键约束：源码必须被容器看到

`docker_gate.py` 会校验宿主源码目录与容器内挂载点是**同一个 inode**，
防止「在 A 目录构建、在 B 目录测试」。本服务器的容器把 `/home` 整体挂进去，
所以两个源码树天然满足：

```bash
# 宿主与容器内 dev:ino 必须一致
stat -c "%d:%i" /home/s00653124/workspace/tilelang
docker exec s00653124-tilelang-hivm stat -c "%d:%i" /home/s00653124/workspace/tilelang
```

若换成只挂部分目录的容器，这一步会失败并明确报错。

### 自检

```bash
cd /path/to/AgentGates
.venv/bin/python -B agent_gates/adapters/tilelang-hivm/scripts/docker_gate.py --check-config
# 期望：exit 0，末行 configuration ready: ... container=... user=... hivmCompiler=...
```

---

## 二、ascendnpu-ir：没有 `env.sh`，靠单元环境

它只需要三个容器身份变量，且**必须**放在 systemd 单元里，因为它不读文件：

```ini
Environment=ASCENDNPU_CONTAINER_NAME=s00653124-tilelang-hivm
Environment=ASCENDNPU_CONTAINER_USER=s00653124
Environment=ASCENDNPU_CONTAINER_HOME=/home/s00653124
```

安装时通过 `extra_env` 写入：

```python
from agent_gates import systemd_units
from agent_gates.config import load

c = load()
systemd_units.install(
    c.worker.python,
    timeout_seconds=7200,
    extra_env={
        "ASCENDNPU_CONTAINER_NAME": "s00653124-tilelang-hivm",
        "ASCENDNPU_CONTAINER_USER": "s00653124",
        "ASCENDNPU_CONTAINER_HOME": "/home/s00653124",
    },
)
```

可选键（不设则用默认）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `ASCENDNPU_BUILD_SUBDIR` | 构建目录名 | `build` |
| `ASCENDNPU_NINJA_JOBS` | 并行度 | 容器内 `nproc` |
| `ASCENDNPU_CANN_SETUP` | CANN 环境脚本 | 空 |
| `ASCENDNPU_CCEC_BIN` | ccec 路径 | 空 |

自检：

```bash
cd /path/to/AgentGates
ASCENDNPU_CONTAINER_NAME=s00653124-tilelang-hivm \
ASCENDNPU_CONTAINER_USER=s00653124 \
ASCENDNPU_CONTAINER_HOME=/home/s00653124 \
.venv/bin/python -B agent_gates/adapters/ascendnpu-ir/scripts/docker_gate.py --check-config
# 期望：exit 0，"configuration valid"
```

---

## 三、常见错误与含义

| 报错 | 原因 | 处理 |
|---|---|---|
| `missing required environment variable ASCENDNPU_CONTAINER_NAME` | 单元里没写这三个变量 | 重新 `install()` 并带 `extra_env` |
| `env.sh does not define TILELANG_HIVM_CONTAINER_NAME ...` | `env.sh` 不存在或缺键 | 按上面模板创建 |
| `container ... is not running` | 容器名对但没启动 | `docker start <name>` |
| `source ... does not match the bind mount` | 源码未挂进容器或挂载点不同 | 检查容器挂载 |
| `missing required environment variable TILELANG_HIVM_DEVICE_ID` | NPU Adapter 未被 AgentGates 调度（直接手跑时需自行指定） | 走 `submit` 而非手跑 |

注意最后一条：手动执行 `npu_gate.py` 需要自己给 `TILELANG_HIVM_DEVICE_ID`；
通过 AgentGates 提交时由本地配置解析，工具调用不能指定设备。
