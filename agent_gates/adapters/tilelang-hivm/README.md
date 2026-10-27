# TileLang HIVM adapter

`tilelang-hivm-build` runs fixed `ninja -C build -j <jobs> tilelang`. `tilelang-hivm-regression` depends on it and runs exactly `testing/ascend/target/test_tilelang_ascend_target.py` and `testing/ascend/cache/test_tilelang_ascend_binary_cache.py` through pytest with JUnit reporting. Inner timeouts are 840 and 900 seconds; outer limits are 900 and 960 seconds.

Regression requires a real zero process exit, a positive collected count, at least one non-skipped test, a native library set, and SHA256 agreement between the container's imported module/native libraries and host files. Zero tests, all skips, altered libraries, missing reports and timeouts cannot pass.

Both TileLang adapters read one strict data file selected with `--env-file`, mapped to `TILELANG_HIVM_ENV_FILE`. Its container identity is authoritative over shell exports. CANN loads before paired Conda setup/environment. Optional HIVM compiler and fixed target are explicitly forwarded into Docker; source bind mounts and inode identity are verified before execution.

The base adapter never dispatches the separate NPU adapter. `full-build.sh`, `setup-python-env.sh`, and `docker/` are manual support files, never gate dependencies. The provisioning helper can install packages and query device availability, so invoke it only deliberately with approved local configuration.

See [adapter configuration](../../../../docs/adapters.md). Scripts originate in PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`; changes only adapt packaged configuration discovery. No upstream root license was found; no license grant is inferred.
