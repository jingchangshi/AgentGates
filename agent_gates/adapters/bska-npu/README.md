# BiShengKernelAgent NPU smoke adapter

This Adapter proves that a deployed BiShengKernelAgent (helix) runtime works on
a real Ascend NPU. It contributes one Recipe, one Gate and one machine file. It
does not share the TileLang or TLA files.

## What it proves

Inside the configured container, on the explicitly authorized device, the Gate
sources CANN, uses the configured conda environment and proves that:

* torch, torch_npu and triton import;
* an NPU is visible and a device allocation succeeds;
* a real Triton vector-add kernel compiles, runs, synchronizes and matches the
  Torch reference numerically;
* node, claude and helix are available inside the container;
* the exact `triton_ascend` build named in the machine file is installed, and
  when a version is pinned the running interpreter matches it.

It does not drive a helix optimize round; that is a deliberate, more expensive
gate for later work.

## Gate requirements

The Gate fails unless the process exits zero and the receipt records every
observation as true: containerRunning, cannSetupReadable, condaEnvPresent,
torchImported, torchNpuImported, tritonImported, npuAvailable,
deviceCountPositive, tensorOnDevice, numericalOk, synchronizeOk, nodeAvailable,
claudeAvailable, helixAvailable, tritonAscendInstalled,
tritonAscendVersionMatch. A missing observation is false, never assumed.
The receipt also records `tritonAscendVersion`, `tritonAscendExpectedVersion`
and `tritonAscendVersionPinned` for review.

## Device selection

This is a device Adapter. The Gate requires an explicitly authorized device
through --device-id or BSKA_NPU_DEVICE_ID; it never discovers or substitutes a
card. It exports ASCEND_RT_VISIBLE_DEVICES for the in-container process. The
container is created with all local NPU device nodes and --network host.

## Configuration

One strict NAME=value file, env.sh, beside env.example.sh. It holds the complete
CANN set_env.sh, the conda environment, the container identity and executable
paths. Pass an absolute path with --env-file (mapped to BSKA_NPU_ENV_FILE) or
keep the git-ignored env.sh in place. The Gate parses it as data and never
executes it.

On this host the /home/<user>/CANN/cann tree is incomplete (no libhccl.so and no
Ascend950PR operator package), so env.sh points BSKA_NPU_CANN_SETUP at the
complete 9.3.0-weekly toolkit under /data/pri/9.3.0-weekly/cann.

## Pinning the triton_ascend build

Vendor `triton_ascend` wheels change the TLA API between daily drops, so an
optimize run is reproducible only when one exact build is named. Two optional
keys in env.sh do that:

* `BSKA_NPU_TRITON_WHEEL` — absolute CONTAINER path to one vendor wheel.
  `scripts/setup-env.sh` installs it with `--force-reinstall --no-deps`, and
  `--check-config` refuses a run if the file is not readable inside the
  container. The wheel is already reachable through the `/data:/data` bind
  mount, for example
  `/data/daily/<date>/triton_ascend-<version>-cp311-cp311-linux_x86_64.whl`.
* `BSKA_NPU_TRITON_ASCEND_VERSION` — the exact distribution version string,
  the same value as `importlib.metadata.version("triton-ascend")`.

The Gate reads the installed version at run time and records
`tritonAscendInstalled`, `tritonAscendVersionMatch`, `tritonAscendVersion`,
`tritonAscendExpectedVersion` and `tritonAscendVersionPinned`. With a version
pin set, a mismatch fails `tritonAscendVersionMatch`; the receipt also shows
both version strings. With no pin, the Gate does not enforce a version and
records `tritonAscendVersionPinned=false`, so an unpinned PASS cannot be
mistaken for a verified one.

To switch builds, set both keys and re-run `scripts/setup-env.sh`; the next
Gate run then proves the running interpreter matches.

## Container and provisioning

docker/ supplies the deployment files:

    cd docker
    cp config.example.env config.env      # set identity, mounts, devices
    ./create-container.sh --config ./config.env --dry-run
    ./create-container.sh --config ./config.env

Then provision node, claude, the conda environment and helix once:

    bash scripts/setup-env.sh --base-env <validated-env-name> \
        --helix-repo /home/<user>/workspace/BiShengKernelAgent

The Gate never calls create-container.sh or setup-env.sh; they are deliberate
operator actions.

## Usage

    agent-gates doctor --adapter bska-npu --source <bska-checkout> \
        --env-file <abs>/env.sh
    agent-gates run --adapter bska-npu --source <bska-checkout> \
        --env-file <abs>/env.sh --device-id 0 --all
    agent-gates verify --run-id <RUN_ID>
