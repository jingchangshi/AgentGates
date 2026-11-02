# TLA NPU end-to-end adapter

This Adapter runs one fixed script, `test_tla.py`, on a real Ascend device and
decides whether it passed. It is separate from the two TileLang adapters: it
contributes its own Recipe, its own Gate and its own machine file, and running
`--all` on `tilelang-hivm` never invokes it.

## What it proves

`test_tla.py` is a 3-in-1 end-to-end check of the `tla` namespace that the
vendor `triton_ascend` wheel adds to Triton:

* it builds a `cast_back` kernel with `@tla.jit`, using `tla.alloc_buffer`,
  `tla.view`, `tla.copy`, `tla.scope.simd()` and `tla.simd.*`;
* it launches it on an NPU with a real fp8 input and checks the result against a
  Torch reference, printing `CAST_BACK OK`;
* it runs the same semantics as a plain `@triton.jit` kernel and checks that
  too, printing `TRITON_CAST_BACK OK`.

A PASS therefore means this machine compiled and executed a real
Triton-on-Ascend kernel through the TLA namespace and got correct numbers, and
that the community Triton backend still works alongside it. It does **not** mean
the whole TLA surface is covered, and it is not a compiler test suite.

## What the Gate requires

The Gate fails unless the test process exits zero **and** the receipt records
that:

* both markers above appear in the test's own stdout, each distinguished from the
  other (the plain-Triton marker contains the TLA marker as a substring, so a
  plain substring test would credit both routes from one line);
* tensors really moved to the device - counted in the process that ran the test,
  from the `device='npu'` factory keyword and `Tensor.npu()`;
* the device actually allocated memory (peak allocation > 0);
* the compiler produced at least one kernel image (`.npubin`). A `.o` is counted
  and reported but not required: the Ascend backend does not emit one, so
  requiring it would be unsatisfiable rather than strict;
* exactly one device is visible to the process;
* device synchronization succeeded, with the original error text carried into the
  receipt when it does not.

The device UUID, its name and the visible-device echo are recorded in the receipt
for a reader, and are deliberately NOT gated: the echo is a value this Gate itself
exported, and both cards report the same UUID on this host, so gating on either
would be true by construction. What the Gate proves is that a compiled TLA kernel
ran on the single card the operator named - not that it independently re-identified
that board. Board serials are host-side facts here; the container's `npu-smi`
cannot load `libc_sec.so`.

Missing data is false, never assumed. A run that produces no output, an
unavailable device, an all-skipped suite and a failed synchronization cannot
pass.

## Device selection

The Gate requires an explicitly assigned `TLA_NPU_DEVICE_ID` or CLI
`--device-id`; it never discovers, chooses or substitutes a device. Root
execution inside the container preserves the upstream `uda_occupy_dev_by_ns`
driver workaround — root does not authorize a device.

Occupancy is reported, not enforced: on this class of machine a card is shared,
and refusing because someone else is on it would block work for a reason
unrelated to the code under test. `TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE=1`
(forwarded from `devices.require_exclusive_device`) turns a held card into
`BLOCKED_ENV`, which is honest because the test never ran.

## Configuration

One strict `NAME=value` file, `env.sh`, beside `env.example.sh`. It holds
CANN, Conda, the container identity and the provisioning values. Pass an
absolute path with `--env-file` (mapped to `TLA_NPU_ENV_FILE`), or keep the
git-ignored `env.sh` in place. The Gate parses the file as data and never
executes it. Unknown keys, duplicate keys and malformed lines fail.

`TLA_NPU_DEVICE_ID` is deliberately not in that file: it authorizes one run,
not a machine.

## Container

`docker/` supplies the deployment files. Build and create the container with:

    cd docker
    docker build -t tla-npu-dev:24.04 .
    cp config.example.env config.env   # set identity, mounts, devices
    ./create-container.sh --config ./config.env --dry-run
    ./create-container.sh --config ./config.env

The image provides the base toolchain only (Ubuntu 24.04, GCC, Python, git). It
deliberately does not contain CANN, Conda, the Python environment or any wheel:
CANN and Conda are bind-mounted from the host, and the environment is built by
`scripts/setup-python-env.sh`. Unlike the TileLang image it installs no
LLVM/clang, cmake or ninja, because the vendor Triton build ships its own prebuilt
LLVM and BishengIR binaries and nothing here compiles C++.

The Gate never calls `create-container.sh` or `setup-python-env.sh`; they are
deliberate operator actions, not gate steps.

## Provisioning

Inside the container, as the container user:

    bash adapters/tla-npu/scripts/setup-python-env.sh

It reads the same `env.sh` and installs the vendor `torch`, `torch_npu` and
`triton_ascend` wheels into `TLA_NPU_CONDA_ENV`. Dependencies are resolved, not
skipped: the Triton wheel pins exact versions, and an environment with different
ones is not the environment the Gate was verified against. Its final step proves
`triton.language.extra.tla` exports the API the Gate test uses and that an NPU is
visible, and it fails rather than leaving a half-usable environment behind.

### The Triton wheel is pinned by name, and the name matters

`TLA_NPU_TRITON_WHEEL` names the exact build to install; discovery is only a
fallback. This is not tidiness. The TLA API is not stable across vendor builds,
and the build that merely sorts newest is not always the one that can run the
test:

* `triton_ascend-3.6.0.post20261030+gita414a5cb` ships a `tla` package that
  cannot compile a TLA kernel. It has no `tla.jit`, so `@tla.jit` raises
  `AttributeError` immediately; and its native library exports only 9 of the 34
  `create_tla_*` / `create_simd_*` builder methods its own `tla/builder.py`
  tries to attach, so aliasing `tla.jit` to `triton.jit` merely moves the failure
  into native code (`no attribute 'create_tla_alloc'`). It cannot be rescued by a
  shim or by pairing it with another wheel. This is a property of the wheel, not
  of the install. Note it *does* ship `tla.scope.simd`/`scope.cube`, which come
  from `tla/vf.py`; the missing pieces are `jit` and the native bindings.
* `triton_ascend-3.6.0.dev20261030+git0c9d99ca`, built the same day, contains
  the complete surface and runs the test to `CAST_BACK OK` /
  `TRITON_CAST_BACK OK` with zero mismatches.

The environment check refuses a wheel whose `tla` package lacks the entry points
the test needs, so provisioning cannot quietly produce an environment that looks
correct until the test fails on an attribute error. If an operator moves to a
newer build, point `TLA_NPU_TRITON_WHEEL` at it and re-run
`setup-python-env.sh`; the check re-validates the API.

## Two inputs, deliberately separate

The Gate has two inputs and they are not the same directory.

`--source` is the **triton-ascend checkout under test**. That is where TLA is
implemented: `third_party/ascend/language/tla/`, which the build installs into
`triton/language/extra/tla/` inside the environment (`setup.py` maps a backend's
`language/` directory onto `triton.language.extra`). Because it is a Git
checkout, provenance is its Git identity: the recorded head commit plus a hash
over the tracked files.

`TLA_NPU_TEST_DIR` is the **directory holding `test_tla.py`**. It is not part of
the source tree, and the Gate will not accept it as one.

Both are mapped into the container through the bind mounts and both sides are
proven to be the same directory by comparing device/inode pairs before anything
runs, so the bytes tested are the bytes on disk. Neither directory is entered as
a working directory: the runner executes from its own scratch directory, because
CANN deposits crash dumps into the process working directory.

## Which TLA code actually ran

The test imports the **installed** `tla` package, not the checkout. Recording
only the checkout would describe source that may not be what executed, so the
receipt also carries `tlaImplementationSha256` and `tlaImplementationFileCount`
(a hash over the installed `tla/*.py` files), `tritonVersion`, and `tlaHasDma`.
These are recorded, not gated: the counter observations decide the verdict. Their
purpose is to let a reader tie a PASS to the code that produced it. A version
string alone cannot do that, since a wheel can be rebuilt at the same version.

`TLA_NPU_CACHE_DIR` names a dedicated `TRITON_CACHE_DIR`, and the Gate refuses a
value inside either input tree: the cache is written to, and both trees are
hashed. Letting a JIT cache land in one of them is how a provenance hash moved
between two runs of the same Gate.

See [adapter configuration](../../../../docs/adapters.md) for the general
adapter, environment-file and receipt conventions.