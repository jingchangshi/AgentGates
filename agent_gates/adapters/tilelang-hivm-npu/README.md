# TileLang NPU sanity adapter

This separate adapter runs the fixed SIMD selection of `examples/ascend/test_vecadd.py` with `pytest -k simd`. It requires an explicitly assigned `TILELANG_HIVM_DEVICE_ID` or CLI `--device-id`; it never chooses another device. The base `tilelang-hivm` adapter does not invoke it.

The container process runs as root, preserving the upstream driver workaround for `uda_occupy_dev_by_ns`. Root execution does not authorize a device: obtain allocation and confirm that it is free before running the gate.

The gate requires collected, passing tests; actual tensor transfers; observed numerical assertions; positive device allocation; the authorized physical-device mapping before and after; and successful synchronization. Missing reports, CPU fallback, all skips, device mismatches and synchronization failures cannot pass. The scope is SIMD/AscendC/BiSheng vector-add sanity, not complete HIVM code generation coverage.

Both TileLang adapters read the same strict data configuration selected with `--env-file`, mapping to `TILELANG_HIVM_ENV_FILE`. The default is `tilelang-hivm/env.sh` beside the sibling adapter template. The NPU gate accepts the shared configuration's container-user key but executes as root.

See [adapter configuration](../../../../docs/adapters.md) for configuration and gate commands. Scripts originate in PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`; local changes only adapt configuration discovery. No upstream root license was found; no license grant is inferred.
