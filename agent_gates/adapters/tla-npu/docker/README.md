# Manual container support

These deployment files create the container the Gate addresses. The Gate never
calls them; they are deliberate operator actions.

The Dockerfile supplies Ubuntu 24.04, distribution GCC/G++, distribution Python,
git and the small set of runtime libraries Torch, torch_npu and the CANN loader
path need. CANN, Conda, the source directory and the vendor wheels are separately
mounted. Unlike the TileLang image it installs no LLVM/clang, cmake or ninja: the
vendor Triton build ships its own prebuilt LLVM and BishengIR binaries inside its
wheel, so nothing in this container compiles C++ and a second toolchain would only
be an unreviewed one.

Build it and create the container:

    docker build -t tla-npu-dev:24.04 .
    cp config.example.env config.env
    ./create-container.sh --config ./config.env --dry-run
    ./create-container.sh --config ./config.env

`create-container.sh` sources a trusted Bash `config.env`. Review it before any
deliberate invocation; it is distinct from the strict data-only gate machine file.
Set the actual identity, permissions, mounts and explicitly assigned device
visibility, and keep the private file out of version control (`*.env` is
git-ignored, and the all-empty template is re-included by name). `--dry-run`
validates paths and prints the command without creating a container, although it
may query Docker availability.

The helper refuses an existing container and missing required mounts. Its
UID/GID/HOME check protects access to bind-mounted source and HOME. The
entrypoint is byte-identical to the TileLang adapter copy and manages the
container account/readiness only. It never provisions CANN, Conda or Python and
never changes host HOME ownership.

Inherited defaults include privileged mode. Device visibility has no default: an
omitted or empty `ASCEND_VISIBLE_DEVICES` exports an empty
`ASCEND_RT_VISIBLE_DEVICES`, which the runtime reads as "no filtering". A literal
default list was removed deliberately — it let a configuration file that merely
left the line out expose cards nobody allocated. Name devices explicitly for your
allocation. These settings do not authorize a device and are never used to choose
a gate's device: the Adapter still requires a separate explicit device ID.

After the container exists, build the Python environment once with
`../scripts/setup-python-env.sh` (see the adapter README and the machine
`../env.sh`).

See [adapter configuration](../../../../docs/adapters.md).
