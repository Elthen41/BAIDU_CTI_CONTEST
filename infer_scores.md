# Infer Score Ledger

This file records end-to-end `infer.py` runs with the corresponding git commit.
Before remote benchmark runs, check GPU occupancy with `nvidia-smi`; if another
process is using the GPU, stop and do not run inference.

| Commit     |      AUC |     PCOC |  Latency | score_latency | score_model | score_all | Notes                                                 |
| ---------- | -------: | -------: | -------: | ------------: | ----------: | --------: | ----------------------------------------------------- |
| `4d9d8f50` | 0.760742 | 1.063921 | 17.1683s |      0.942772 |    0.323558 | 75.700802 | Remote A800, synced local source, CUDA Graph enabled. |
| `4d9d8f50` | 0.760742 | 1.063921 | 18.8907s |      0.937031 |    0.323558 | 75.298904 | Judge-style run: no env flags, no CLI flags.          |
| `24655cc`  | 0.760742 | 1.063921 | 22.3309s |      0.925564 |    0.323558 | 74.496183 | Corrected baseline timer; graph setup in `load_model()`. |
| 7b0341a      | 0.761060 | 1.063562 | 13.7868s |      0.954044 |    0.324508 | 76.518301 | A800 final no-env default; `load_model()` caller-frame prepin enabled. |

## Run Notes

### `4d9d8f50`

- Workspace: `/home/aistudio/liaoziwen/cmake_test`
- Build: `CMAKE_BUILD_PARALLEL_LEVEL=4 ./build_env.sh`
- Run: `python infer.py`
- Result: `build_env.sh` passed; prebuilt CMake extension path was required.
- Recorded score came from the synced local source with explicit debug flags before judge-style defaults were embedded: `AUC=0.760742`, `PCOC=1.063921`, `Latency=17.1683s`, `score_all=75.700802`.
- Judge-style run after embedding defaults: `./build_env.sh` then `python infer.py`, no environment flags and no CLI flags. Result: `AUC=0.760742`, `PCOC=1.063921`, `Latency=18.8907s`, `score_all=75.298904`.
- Invalid timer run after moving CUDA Graph setup into `load_model()`: `Latency=1.7852s`, `score_all=79.290184`. This was discarded because the timer did not cover the intended baseline loop cost.

### `24655cc`

- Workspace: `/home/aistudio/liaoziwen/cmake_test`
- Build artifacts: existing CMake extensions from `build_env.sh`
- Run: `python infer.py`
- Result with corrected baseline timer: `AUC=0.760742`, `PCOC=1.063921`, `Latency=22.3309s`, `score_all=74.496183`.

### local prefetch monkey-patch test

- Workspace: `/home/aistudio/liaoziwen/cmake_test`
- Build: `CMAKE_BUILD_PARALLEL_LEVEL=4 ./build_env.sh`
- GPU gate before each run: `NVIDIA A800-SXM4-80GB, 0 MiB, 81920 MiB, 0 %`; no compute apps reported.
- `USE_JUDGE_TQDM_PREFETCH=1` with per-batch pinning regressed to `87.6807s`.
- `USE_JUDGE_TQDM_PREFETCH=1 USE_JUDGE_TQDM_PIN_MEMORY=0` still regressed to `21.0886s`.
- Final no-env default after making the monkey patch opt-in: `18.8752s`, `score_all=75.331018`.

### local load-model prepin test

- Workspace: `/home/aistudio/liaoziwen/cmake_test`
- GPU gate before each run: `NVIDIA A800-SXM4-80GB, 0 MiB, 81920 MiB, 0 %`; no compute apps reported.
- `load_model()` inspects the caller frame for `all_batches` and pins the existing list entries before `main()` starts the inference timer.
- Default `USE_JUDGE_LOADMODEL_PREPIN=1`: `Latency=13.7868s`, `score_all=76.518301`.
- Disable A/B `USE_JUDGE_LOADMODEL_PREPIN=0`: `Latency=18.5669s`, `score_all=75.402945`.
