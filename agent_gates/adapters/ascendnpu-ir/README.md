# AscendNPU-IR adapter

> [!WARNING]
> **Docker route quarantined (`DOCKER_MODE_DISABLED`)**: `ascend-build` and
> `ascend-regression` are currently disabled and isolated (see
> [`docs/docker-mode-quarantine.md`](../../../docs/docker-mode-quarantine.md)).
> Active runs use `ascend-build-host` and `ascend-regression-host`.

`ascend-build` runs the fixed Ninja targets `bishengir-opt bishengir-compile install-bishengir-publish-products`. `ascend-regression` depends on that build and runs `check-bishengir`. A real zero exit is required. `ascend-build-host` and `ascend-regression-host` run the same two steps on the host, with no Docker in the path.

## Two routes

This Adapter offers the same Gate by two routes. They are separate gates with separate recipes, and neither route depends on the other: `Manifest.order(['ascend-regression-host'])` expands to `['ascend-build-host', 'ascend-regression-host']`, and the docker pair expands to `['ascend-build', 'ascend-regression']`.

| Route | Gates | Needs |
|---|---|---|
| docker | `ascend-build`, `ascend-regression` | a running build container, the Docker socket, the bind mounts `docker/config.env` declares |
| host | `ascend-build-host`, `ascend-regression-host` | the toolchain on the host, in the effective PATH |

Use the docker route when the container already exists and is the machine that builds this tree: it is the route whose result the operator has been reading, and its payload is unchanged. Use the host route when there is no container to enter - a machine without Docker socket access is not a degraded Adapter, it is an unusable one - and the question ("does this tree still build, does `check-bishengir` still pass?") is exactly what does not need a container.

The host route is not a second implementation of the Gate. It is a wrapper that loads the pinned sibling `scripts/docker_gate.py` and binds its parser, build-tree probe, timeout chain and from-zero full-build command, so only the transport differs. What it adds is a preflight of the *host* toolchain, which Docker mode inherits from its image and never has to check: `ninja >= 1.12` (`CMakeLists.txt:6-13`), `cmake >= 3.28` (`CMakeLists.txt:2`), and `bash`, `timeout`, `nproc`, `clang`, `clang++`, `g++`, `python3`, `ld.lld`, `git` - plus CANN defining `ASCEND_HOME_PATH` when `ASCENDNPU_CANN_SETUP` is set. The check resolves every tool from the effective PATH, after CANN and Conda, and prints the path beside the version. A from-zero host build additionally needs executable `ccec` and `llvm-link` in `ASCENDNPU_CCEC_BIN` and populated `third-party/llvm-project` and `third-party/shmem`; the build check reports all of that at t=0 rather than after a long configure, and an existing build tree needs none of it.

`run --all` now selects every gate in the Adapter, which is both routes. A host-only operator therefore names its route: `--gate ascend-build-host --gate ascend-regression-host`, and `doctor` takes the same selection.

## One machine file

Everything this Adapter knows about the machine comes from one file:
`agent_gates/adapters/ascendnpu-ir/env.sh`, or the absolute path in
`ASCENDNPU_ENV_FILE` / `--env-file`. The launching environment is never a
configuration source. An export the file does not define is refused, and an
export that disagrees with the file is refused, so one run cannot use two
values. Sourcing `env.sh` and then running a Gate still works, because the
export then equals the file.

| Key | Meaning | Default |
|---|---|---|
| `ASCENDNPU_CONTAINER_NAME` / `_USER` / `_HOME` | container the docker route enters | none; all three required by that route only |
| `ASCENDNPU_CANN_SETUP` | CANN `set_env.sh` | empty |
| `ASCENDNPU_CONDA_SH` / `_CONDA_ENV` | conda.sh and environment | empty; set both or neither |
| `ASCENDNPU_CCEC_BIN` | ccec compiler bin directory | empty; needed for a from-zero build |
| `ASCENDNPU_BUILD_SUBDIR` | build directory name | `build` |
| `ASCENDNPU_NINJA_JOBS` | `ninja -j` value | the route's CPU count |

The file is data: strict `NAME=value`, comments and blank lines. It is parsed,
never executed, so substitutions do not happen. Unknown keys, duplicate keys,
malformed lines and unbalanced quotes are refused.

AgentGates reads the same file when it takes its `build:<dir>` lease, so a lock
on `build` cannot cover a Gate writing `build-release`.

## Two modes

`ascend-build` inspects the build directory inside the container before running:

| Tree state | Behaviour |
|---|---|
| `build.ninja` present | incremental `ninja -C <dir> -j <jobs> <targets>` |
| neither `build.ninja` nor `CMakeCache.txt` | from-zero configure plus full build through the pinned `scripts/full-build.sh`, then the products are checked |
| `CMakeCache.txt` without `build.ninja` | refused; the Adapter never deletes a tree it did not create |

The from-zero branch runs exactly the operator's command, with `-t` meaning
`--build-bishengir-template`, `--build ./<ASCENDNPU_BUILD_SUBDIR>` and
`-j <ASCENDNPU_NINJA_JOBS>`. It refuses `-r`/`--rebuild`, because that flag makes
`build-tools/build.sh` delete the build directory. It never overwrites an
existing `build.log`: the log becomes `build-<UTC>.log` beside it, and inside the
Gate it goes to stdout only, so a Gate writes no file into the source tree.

`ascend-regression` runs an existing tree only. Its container script carries no
full-build branch at all, so an unbuilt checkout cannot report PASS for a test
that never ran.

The host route decides the same three states the same way, in Python, before it
runs anything. Its regression configuration check succeeds (exit 0) even when no
build tree exists yet - `doctor` on a new machine is exactly that case - and
merely reports the absent `build.ninja`; the regression recipe itself is what
refuses a from-zero tree.

## Exit codes

A configuration refusal is 2 (`BridgeError`). A from-zero build failure or an
inner timeout is 3: the shared Generator's own `fail` wraps both
(`docker_gate.py:666-668`). The incremental path preserves `timeout`'s 124
unchanged. Nothing broader is a declared blocked exit code; these are the
observed meanings of the codes the two scripts return, not a claim that every
non-zero code means "blocked".

## Timeout chain

Incremental 840s and full 21600s container-side, recipe outer bound 23400s,
worker `RuntimeMaxSec` 23700s. The worker value is derived from the installed
manifests (`systemd_units.backstop_seconds()`), not fixed: an installer that
passes a smaller value is refused.

Both host recipes use the same values, because they run the same Generator: 840s
incremental, 21600s from-zero, 23400s outer for `ascend-build-host`; 1140s inner
and 1200s outer for `ascend-regression-host`.

## Other behaviour

The bridge derives the source path from Docker bind mounts, rejects ambiguous or
missing mappings, and compares host/container device and inode before running the
build. Configuration checks run on the host without Docker, for both routes: the
docker checks never spawn `docker`, and the host checks never configure, never run
cmake and never write into the source tree. New host messages carry the
`ascendnpu-ir-host-gate` prefix; the shared Generator keeps its own
`ascendnpu-ir-docker-gate` prefix in the script it generates, because rewriting
it would be the fork the wrapper exists to avoid. The committed
initialization shell and the full-build helper are inlined into the container
command; there is no runtime dependency on Harness.

See [adapter configuration](../../../../docs/adapters.md). Scripts originate in PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`. Changes add packaged configuration discovery, host-only configuration checking, the from-zero build branch, and the host route that runs the same Gate without Docker, preserving build commands and checks. No upstream root license was found; no license grant is inferred.
