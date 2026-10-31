# Triton-on-Ascend (TLA) development container image.
#
# This image is modelled on the "tilelang-hivm" image next door and is used the
# same way: it provides the *base toolchain* only, never CANN, Conda, a Python
# environment or any wheel.
#
#   * CANN is machine-specific and version-specific, so it is bind-mounted from
#     the host into the running container, not baked in.
#   * Conda is the operator's own installation, bind-mounted the same way.
#   * The Triton build under test comes from a vendor wheel installed into a
#     Conda environment inside the container; see ./README.md and
#     ../scripts/tla_gate.py.
#
# Deliberately NOT installed, compared with the tilelang-hivm image:
#
#   * clang/lld/LLVM. The Triton build under test ships prebuilt LLVM and
#     BishengIR binaries inside the wheel (`triton/_C/*`, and
#     `triton/backends/ascend/bishengir/bin/*`), so no C++ is compiled here and
#     a host LLVM would only be a second, unreviewed toolchain.
#   * cmake and ninja. Nothing in this container configures a build tree; the
#     Triton wheel is a binary distribution. Installing them would suggest a
#     source build that this container is not used for.
#
# Everything that IS installed is required by one of: the entrypoint (passwd,
# util-linux), the interpreter the Conda environment is created for (python3,
# python3-dev, python3-venv), the Torch/Torch-NPU runtime wheels and the CANN
# loader path (libgomp1, zlib, libstdc++, numactl), or the gate script's own
# probes (bash, coreutils, file, git).
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu archive mirror. Plain HTTP, not HTTPS, on purpose: the base image has no
# ca-certificates yet, and an HTTPS mirror therefore fails TLS verification while
# `apt-get update` still exits 0, which would let the build continue with an
# empty package index. The distribution signing key still authenticates every
# package, so a wrong or hostile mirror cannot install one.
ARG UBUNTU_MIRROR=http://repo.huaweicloud.com/ubuntu/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Noble uses the deb822 /etc/apt/sources.list.d/ubuntu.sources, not the legacy
# sources.list, so the mirror is rewritten there. The rewrite is guarded: if a
# future base image drops that file the build fails here rather than silently
# installing from a different mirror. `apt-get update` is not trusted to fail on
# its own -- it warns and exits 0 when an index is unreachable -- so the rewrite
# and the resulting index are both verified explicitly.
RUN set -eux; \
    sources=/etc/apt/sources.list.d/ubuntu.sources; \
    [ -f "$sources" ] || { echo "expected $sources to exist" >&2; exit 1; }; \
    sed -i \
        -e "s|^URIs: http://archive.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        -e "s|^URIs: http://security.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        -e "s|^URIs: https://archive.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        -e "s|^URIs: https://security.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        "$sources"; \
    grep "^URIs:" "$sources"; \
    [ "$(grep -c "^URIs: ${UBUNTU_MIRROR}$" "$sources")" -eq 2 ] \
        || { echo "mirror rewrite did not cover both suites" >&2; exit 1; }; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends \
        bash coreutils ca-certificates curl wget gnupg git \
        make gcc g++ build-essential pkg-config file \
        python3 python3-dev python3-venv python3-pip \
        passwd util-linux \
        locales \
        libgomp1 zlib1g-dev libzstd-dev libedit-dev libffi-dev \
        libssl-dev libxml2-dev libncurses-dev liblz4-dev \
        libnuma1 numactl; \
    # Prove the index actually came from the mirror rather than an empty cache:
    # apt names the list files after the mirror host, and the indexes are lz4
    # compressed, so the glob must allow a suffix. An unreachable mirror leaves
    # this directory empty but still exits 0 above.
    ls /var/lib/apt/lists/*_Packages* >/dev/null \
        || { echo "apt package index is empty after update" >&2; exit 1; }; \
    sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen; \
    locale-gen; \
    rm -rf /var/lib/apt/lists/*

# Prove the toolchain is usable with no activation, because the gate script's
# non-interactive probes are bare `bash`/`stat` calls inside `docker exec`.
RUN set -eux; \
    command -v bash; bash --version | head -1; \
    command -v python3; python3 --version; \
    command -v gcc; gcc --version | head -1; \
    command -v git; git --version

# The entrypoint runs as root so it can create the configured UID/GID identity
# inside the container, then the gate runs its own `docker exec`.
COPY entry.sh /usr/local/bin/entry.sh
RUN chmod +x /usr/local/bin/entry.sh

# `--init` is recommended by create-container.sh, so this container gets a
# reaping init without the image depending on tini.
ENTRYPOINT ["/usr/local/bin/entry.sh"]
CMD ["tail", "-f", "/dev/null"]
