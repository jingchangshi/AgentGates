# AscendNPU-IR adapter

`ascend-build` runs the fixed Ninja targets `bishengir-opt bishengir-compile install-bishengir-publish-products`. `ascend-regression` depends on that build and runs `check-bishengir`. A real zero exit is required.

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
| `ASCENDNPU_CONTAINER_NAME` / `_USER` / `_HOME` | container to enter | none; all three required |
| `ASCENDNPU_CANN_SETUP` | CANN `set_env.sh` | empty |
| `ASCENDNPU_CONDA_SH` / `_CONDA_ENV` | conda.sh and environment | empty; set both or neither |
| `ASCENDNPU_CCEC_BIN` | ccec compiler bin directory | empty; needed for a from-zero build |
| `ASCENDNPU_BUILD_SUBDIR` | build directory name | `build` |
| `ASCENDNPU_NINJA_JOBS` | `ninja -j` value | container CPU count |

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

## Timeout chain

Incremental 840s and full 21600s container-side, recipe outer bound 23400s,
worker `RuntimeMaxSec` 23700s. The worker value is derived from the installed
manifests (`systemd_units.backstop_seconds()`), not fixed: an installer that
passes a smaller value is refused.

## Other behaviour

The bridge derives the source path from Docker bind mounts, rejects ambiguous or
missing mappings, and compares host/container device and inode before running the
build. Configuration checks run on the host without Docker. The committed
initialization shell and the full-build helper are inlined into the container
command; there is no runtime dependency on Harness.

See [adapter configuration](../../../../docs/adapters.md). Scripts originate in PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`. Changes add packaged configuration discovery, host-only configuration checking and the from-zero build branch, preserving build commands and checks. No upstream root license was found; no license grant is inferred.
