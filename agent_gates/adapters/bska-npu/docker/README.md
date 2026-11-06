# Manual container support

These deployment files create the container the bska-npu Gate addresses. The Gate
never calls them; they are deliberate operator actions.

The container is created from the sibling base image tla-npu-dev:24.04 (Ubuntu
24.04 with GCC, Python, git and the small runtime libraries Torch/torch_npu and
the CANN loader path need). CANN, Conda, /home and /data are bind-mounted; node
and claude are installed into the running container by ../scripts/setup-env.sh.

    cp config.example.env config.env
    ./create-container.sh --config ./config.env --dry-run
    ./create-container.sh --config ./config.env

config.env sources a trusted Bash file; review it before running. It is
git-ignored. The script refuses an existing container, missing required mounts
and a container identity that cannot read HOME. --network host is required so
the container shares the host loopback proxy at 127.0.0.1:17890.

entry.sh is byte-identical to the sibling adapter copy; it manages only the
container account/readiness and never provisions CANN, Conda, node or claude.
