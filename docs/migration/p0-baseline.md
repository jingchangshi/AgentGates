# P0 original execution baseline

Executed on 2026-10-27 using original digest-verified adapter scripts at PiEngineeringHarness HEAD `9ca04ba6814154a778ecf30168c793d73352e643`. Remote HEAD was independently queried and matches the requested baseline. All three installed adapter manifests and pinned files match the source checkout byte-for-byte. No source checkout was modified.

The original Harness task/agent state was intentionally bypassed as authorized: the recorder executes the manifest's exact recipe argv/CWD and configCheck, then independently compares the original evaluator's conditions. This is an original-recipe baseline, not a claim that a complete Harness task session ran. The local recorder and full results are in ignored `.evidence/run_old_baseline.py` and `.evidence/p0-baseline/`.

| Gate | Result | Exit | Timeout | Seconds | Test evidence |
| --- | --- | --- | --- | ---: | --- |
| `ascend-build` | PASS | 0 | no | 80.670 | Original three Ninja targets completed |
| `ascend-regression` | PASS | 0 | no | 410.713 | 1142 discovered; 1023 passed; 119 unsupported |
| `tilelang-hivm-build` | PASS | 0 | no | 0.615 | Original incremental `tilelang` target |
| `tilelang-hivm-regression` | PASS | 0 | no | 8.727 | 14 collected, 14 passed, 0 failed, 0 skipped; 12 native libraries; required identity observations true |
| `tilelang-hivm-npu-sanity` | BLOCKED_ENV | not run | n/a | n/a | No explicit task-assigned device ID; idle cards are not assignment |

Ascend source HEAD: `bfefe18011152cca671e7e6ddf03793afbacaa2a`. TileLang source HEAD: `6fe6f09182c854e680a5488f4d8461fe5d5a76a6` with five existing modifications preserved. Recorder snapshots before and after the executed gates matched.

The Ascend machine file was absent. The recorder generated an ignored, mode-0600 file containing only verified existing CANN/Conda setup values from TileLang's current machine file. Ascend's runtime container identity variables refer to the same verified existing user container and bind mounts. This is a documented environment adaptation, not an original source change. Ninja jobs were capped at 2 for both builds. Existing build directories were used, as permitted; no images or shared Conda environments were rebuilt/reset.

Each gate directory contains original argv, CWD, config-file digest, source HEAD/snapshot, timestamps, elapsed seconds, actual return code/timeout, stdout/stderr and hashes, parsed observations, acceptance rules and classification in `result.json`. Ascend executable hashes and the TileLang Python module hash were captured. The recorder did **not** persist the individual hashes of the 12 loaded TileLang native libraries: its guessed artifact search paths missed their actual locations. The original bridge reported successful host/container native identity comparison, but that observation is weaker evidence than a retained per-library hash list. This P0 provenance gap is not retroactively repaired using later bytes. P1 separately captures explicit actual native artifacts. Ascend stdout includes raw Ninja/lit output; TileLang stdout is the adapter's JSON observations. Large machine logs remain untracked.

Original manifest SHA256 values:

- AscendNPU-IR: `bce77dee668e7994bd49c11173a6baaa975b7082c1253a8799590539cd75366a`
- TileLang-HIVM: `38ecce402506f1ba5a128d78cde778bb5543eb16d5439147034c01d424881bab`
- TileLang-HIVM-NPU: `6e65d6c29f3c546ba6af737b4cf2d869b0dae5b1d90af23d1578b901a06a3222`

The NPU gate has no real execution evidence in P0 and must not be reported as hardware PASS. Offline fixtures cover its contract and failures separately.
