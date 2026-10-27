# AscendNPU-IR adapter

`ascend-build` runs the fixed incremental Ninja targets `bishengir-opt bishengir-compile install-bishengir-publish-products`. `ascend-regression` depends on that build and runs `check-bishengir`. A real zero exit is required. Inner timeouts are 840 and 1140 seconds; outer limits are 900 and 1200 seconds.

Select machine configuration with `--env-file` (mapped to `ASCENDNPU_ENV_FILE`). The file permits CANN setup, paired Conda setup/environment, and optional CCEC location only. Container name, user and HOME are explicit launch environment values `ASCENDNPU_CONTAINER_NAME`, `ASCENDNPU_CONTAINER_USER`, and `ASCENDNPU_CONTAINER_HOME`. They are not keys in this adapter's machine file.

The bridge derives the source path from Docker bind mounts, rejects ambiguous or missing mappings, and compares host/container device and inode before running the build. It requires an existing `build/build.ninja`. Configuration checks run on the host without Docker. The committed initialization shell is inlined into the container command; there is no runtime dependency on Harness.

`full-build.sh`, `docker/`, and toolchain initialization are manual support files. Gates never provision an environment, configure a fresh tree, create a container, or call the full-build helper.

See [adapter configuration](../../../../docs/adapters.md). Scripts originate in PiEngineeringHarness revision `9ca04ba6814154a778ecf30168c793d73352e643`. Changes add packaged configuration discovery and host-only configuration checking, preserving build commands and checks. No upstream root license was found; no license grant is inferred.
