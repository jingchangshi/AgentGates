# Configuration for the tilefoundry-npu adapter.
#
# This is the single configuration file. One reader uses it and nothing else
# does: scripts/tilefoundry_gate.py.
#
# Create the real file once, on this machine:
#
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# env.sh is git-ignored. It is parsed as DATA and never executed: unknown keys,
# duplicate keys and malformed lines all fail, because a typo that is silently
# ignored is a value the operator believes in and the Gate does not have.
#
# Exporting these names in a shell has NO effect. The Gate reads this file only,
# so a stale export cannot silently override what the machine says it is.

# --- required -------------------------------------------------------------

# The TileFoundry checkout. The adapter REUSES its docker/ascend950pr recipe
# rather than duplicating it, so this path must contain Dockerfile and
# create-container.sh.
TF_REPO=

# Docker image and container names.
IMAGE=tilefoundry-ascend950pr-dev:24.04
CONTAINER_NAME=tf-ascend950pr

# Host CANN root, bind-mounted read-only into the container.
CANN_ROOT=/data/pri/cann-9.2.0

# The Python 3.12 interpreter INSIDE the container: a conda environment under
# /home, which is bind-mounted, so it survives container recreation without a
# reinstall or a re-download.
#
# /opt/tf-venv is DEPRECATED. It is no longer created, mounted or executed, and
# nothing in this adapter may point at it.
IN_CONTAINER_PYTHON=/home/s00653124/.conda/envs/tilefoundry-npu/bin/python

# The account the container runs CPU work as. NPU work must enter as root on
# this platform, which the device Gate does deliberately and records.
CONTAINER_USER=s00653124

# --- device ---------------------------------------------------------------

# DELIBERATELY EMPTY. A device id authorizes one Gate run, not a machine, so it
# is passed per invocation with --device-id or DEVICE_ID. The Gate refuses
# without one and never discovers or substitutes a device.
DEVICE_ID=

# Occupancy is reported, not enforced: this is a shared server, and a busy card
# is not a fact about the code under test. Setting this to 1 turns a held card
# into BLOCKED_ENV (exit 75) -- not a pass, not a failure, no result invented.
TF_REQUIRE_EXCLUSIVE=0

# --- optional -------------------------------------------------------------

# The conda environment for TileFoundry. Lives under /home so it outlives the
# container.
TF_CONDA_ENV=/home/s00653124/.conda/envs/tilefoundry-npu

# The in-container venv holding the NPU environment is created by
# scripts/setup-tf-env.sh. It is python 3.11, not 3.12: every local torch_npu
# wheel is cp311 and torch_npu-2.10.0.post5 declares 'Requires-Dist:
# torch ==2.10.0+cpu', so the pair only exists for 3.11. See the setup script.
TF_NPU_ENV=tilefoundry-npu

# The shared conda installation, used only to create OUR OWN env under
# $HOME/.conda/envs. No other user's environment is read, activated or
# modified. /data is mounted read-only, which is why this is visible.
CONDA_SH=/data/miniconda3/etc/profile.d/conda.sh

# CANN must be sourced BEFORE torch_npu is imported, or torch_npu fails with
# 'ImportError: libhccl.so: cannot open shared object file'. Note the
# ascend-toolkit/ component: /data/pri/cann-9.2.0/set_env.sh does not exist.
CANN_SET_ENV=/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh

# A card another tenant already holds is REFUSED by default (exit 75 BLOCKED_ENV):
# occupancy is read before torch_npu is imported, because a failed aclInit still
# opens the device node. Set this to 1 only to deliberately accept a shared card;
# the run is then recorded as a shared-card result with provisional timings.
TF_ALLOW_BUSY_DEVICE=

# The host proxy script. Its exports are passed into docker exec with -e; direct
# network here runs at ~58 KB/s against ~1.7 MB/s through the proxy.
PROXY_SH=/home/s00653124/set_proxy.sh

# Local vendor wheels. PyPI torch resolves to a CUDA build on this host, so the
# torch + torch_npu pair is installed --no-deps from here.
VENDOR_WHEELS=/home/s00653124/Packages


# Host paths bind-mounted into the container. The workspace parent is what makes
# the repo and the evidence directory visible inside.
TF_WORKSPACE=/home/s00653124/workspace