# AscendNPU-IR long-running development container image.
#
# The image provides the *base toolchain* only. It deliberately does not contain
# CANN, Conda, the AscendNPU-IR source tree, any build products, the Pi Agent or
# this Harness, and it never touches the host NPU Driver:
#
#   * CANN is machine-specific and version-specific, so it is bind-mounted from
#     the host into the running container, not baked in.
#   * Conda is the user's own environment and is bind-mounted the same way.
#   * The source tree comes from a bind mount of the host worktree, so the
#     container compiles exactly the bytes the Host sees, including uncommitted
#     edits. That is what the Adapter proves with its dev/inode check.
#
# The toolchain versions are chosen from the AscendNPU-IR repository's own build
# requirements, not from "whatever Ubuntu 24.04 happens to ship":
#
#   * `clang`/`lld` 18 from Ubuntu noble's own signed archive. `build-tools/
#     build.sh` defaults to `clang`/`clang++`, and the repository is developed
#     against this toolchain; the vendor Bisheng LLVM15 compiler is delivered by
#     CANN at run time, not here.
#   * `cmake` >= 3.28 is a hard floor in `CMakeLists.txt`
#     (`cmake_minimum_required(VERSION 3.28.0)`). Noble ships exactly 3.28.3,
#     which is the floor with no margin, so cmake is installed from pip instead.
#   * `ninja` >= 1.12 is required by the repository documentation; noble's
#     `ninja-build` is 1.11.1, below that, so ninja is installed from pip too.
#
# Only the `cmake` and `ninja` executables are exposed on the standard PATH via
# `/usr/local/bin`. The interpreter stays the distribution `python3`, so the
# Adapter's non-interactive identity probe (`python3 -c ...`) works in a bare
# `docker exec` without any environment activation.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Pinned because pip is the only source of a new enough cmake/ninja on noble.
ARG CMAKE_VERSION=3.31.6
ARG NINJA_VERSION=1.13.0
ARG LLVM_VERSION=18
# Ubuntu archive mirror. The Aliyun mirror is the default because
# archive.ubuntu.com is slow from the usual build hosts. Both the release and
# the security suites are served by the same mirror, and the distribution signing
# key still authenticates every package, so a wrong or hostile mirror cannot
# install one. Override it to use another mirror.
#
# Plain HTTP, not HTTPS, on purpose: the base image has no ca-certificates yet,
# and an HTTPS mirror therefore fails TLS verification while `apt-get update`
# still exits 0, which would let the build continue with an empty package index.
# Ubuntu's own default sources are HTTP for the same reason; upgrade to HTTPS
# after ca-certificates is installed if a deployment requires it.
ARG UBUNTU_MIRROR=http://mirrors.aliyun.com/ubuntu/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ---------------------------------------------------------------------------
# System packages.
#
# * `passwd` provides useradd/groupadd/usermod, which entry.sh needs at start.
# * `util-linux` provides `setpriv`/`runuser`, used to drop privileges.
# * The dev libraries are the ones LLVM/MLIR link against: zlib, zstd, libedit,
#   libffi, libxml2, ncurses, lz4 and OpenSSL.
#
# Noble uses the deb822 `/etc/apt/sources.list.d/ubuntu.sources`, not the legacy
# `sources.list`, so the mirror is rewritten there. The rewrite is guarded: if a
# future base image drops that file the build fails here rather than silently
# installing from a different mirror. `apt-get update` is not trusted to fail on
# its own -- it warns and exits 0 when an index is unreachable -- so the rewrite
# and the resulting index are both verified explicitly.
# ---------------------------------------------------------------------------
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
        make gcc g++ \
        build-essential pkg-config file ccache \
        clang-${LLVM_VERSION} lld-${LLVM_VERSION} libclang-${LLVM_VERSION}-dev \
        libclang-cpp${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-dev \
        python3 python3-dev python3-venv python3-pip \
        passwd util-linux \
        locales \
        libgomp1 zlib1g-dev libzstd-dev libedit-dev libffi-dev \
        libssl-dev libxml2-dev libncurses-dev liblz4-dev; \
    # Prove the index actually came from the mirror rather than an empty cache:
    # apt names the list files after the mirror host, and the indexes are lz4
    # compressed, so the glob must allow a suffix. An unreachable mirror leaves
    # this directory empty but still exits 0 above.
    ls /var/lib/apt/lists/*_Packages* >/dev/null \
        || { echo "apt package index is empty after update" >&2; exit 1; }; \
    sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen; \
    locale-gen; \
    update-alternatives --install /usr/bin/clang clang "/usr/bin/clang-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/lld lld "/usr/bin/lld-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/ld.lld ld.lld "/usr/bin/ld.lld-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100; \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100; \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# cmake and ninja, pinned above the distribution versions.
#
# Installed into an image-private venv so the distribution Python stays
# untouched, then only the two executables are linked onto the standard PATH.
# `--break-system-packages` is therefore not needed.
#
# PYPI_MIRROR is declared here, immediately before its only use, so changing it
# invalidates this layer alone. Declaring it above the system-package layer would
# make a mirror change rebuild that entire layer.
# ---------------------------------------------------------------------------
# The Aliyun mirror is the default because the public PyPI index is slow or
# unreachable from the usual build hosts. The wheel contents are identical and
# pip still verifies them against the index. Override it to use another index.
ARG PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/
RUN set -eux; \
    python3 -m venv /opt/build-tools; \
    /opt/build-tools/bin/pip install --no-cache-dir --disable-pip-version-check \
        -i "${PYPI_MIRROR}" "cmake==${CMAKE_VERSION}" "ninja==${NINJA_VERSION}"; \
    ln -sf /opt/build-tools/bin/cmake /usr/local/bin/cmake; \
    ln -sf /opt/build-tools/bin/ninja /usr/local/bin/ninja; \
    rm -rf /root/.cache; \
    command -v cmake; cmake --version | head -1; \
    command -v ninja; ninja --version; \
    command -v python3; python3 --version; \
    command -v clang; clang --version | head -1

# The entrypoint runs as root so it can create the configured UID/GID identity
# inside the container, then the Adapter runs its own `docker exec -u <user>`.
COPY entry.sh /usr/local/bin/entry.sh
RUN chmod +x /usr/local/bin/entry.sh

# `--init` is recommended by create-container.sh, so this container gets a
# reaping init without the image depending on tini.
ENTRYPOINT ["/usr/local/bin/entry.sh"]
CMD ["tail", "-f", "/dev/null"]
