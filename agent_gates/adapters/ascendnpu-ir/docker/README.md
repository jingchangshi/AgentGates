# Manual container support

These deployment files are retained from PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`. AgentGates gates never call them. The Dockerfile supplies Ubuntu 24.04, Clang/LLVM 18, CMake 3.31.6, Ninja 1.13.0 and distribution Python; CANN, Conda and compiler sources must be supplied separately through reviewed bind mounts.

`create-container.sh` reads a trusted Bash `config.env`, which it sources. This differs from the strict data-only gate machine file. Review container permissions, mounts and explicitly assigned device visibility before deliberate invocation. Start from `config.example.env`; private configuration must stay outside version control. `--dry-run` prints the planned command and creates nothing, although it may query Docker availability.

The helper refuses an existing container, missing required mounts and incompatible UID/GID/HOME. The entrypoint creates/verifies the container account, preserves bind-mounted HOME ownership and writes a readiness marker. It does not install CANN or Conda. The helper never stops or removes an existing container.

The inherited deployment defaults include privileged mode and device visibility `0,1`; they are not an allocation and are not used by the standalone gate CLI. Set reviewed values for your environment. NPU test authorization remains the separate explicit device-ID input of the NPU adapter.

See [adapter configuration](../../../../../docs/adapters.md) for gate setup and scope. No container image build or creation was performed during the offline migration tests.
