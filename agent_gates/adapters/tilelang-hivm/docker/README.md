# Manual container support

These deployment files are retained from PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`. AgentGates gates never call them. The Dockerfile supplies Ubuntu 24.04, distribution GCC/G++, Clang/LLVM 18, CMake 3.31.6, Ninja 1.13.0 and distribution Python. CANN, Conda and TileLang sources are separately mounted. The TileLang image retains GCC/G++ as the default C/C++ compiler.

`create-container.sh` sources a trusted Bash `config.env`. Review it before any deliberate invocation; it is distinct from the strict data-only gate machine file. Start from `config.example.env`, set the actual identity, permissions, mounts and explicitly assigned device visibility, and keep the private file out of version control. `--dry-run` validates paths and prints the command without creating a container, although it may query Docker availability.

The helper refuses an existing container and missing required mounts. Its UID/GID/HOME check protects access to bind-mounted source and HOME. The entrypoint is byte-identical to the Ascend adapter copy and manages the container account/readiness only. It never provisions CANN or Conda and never changes host HOME ownership.

Inherited defaults include privileged mode. Device visibility has no default: an omitted or empty `ASCEND_VISIBLE_DEVICES` exports an empty `ASCEND_RT_VISIBLE_DEVICES`, which the runtime reads as "no filtering". A literal default list was removed deliberately — it let a configuration file that merely left the line out expose cards nobody allocated. Name devices explicitly for your allocation. These settings do not authorize a device and are never used to choose a gate's device. The NPU adapter still requires a separate explicit device ID and never runs from the base adapter.

See [adapter configuration](../../../../../docs/adapters.md). No container image build or creation was performed during the offline migration tests.
