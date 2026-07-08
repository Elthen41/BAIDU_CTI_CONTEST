# Infer Score Ledger

This file records end-to-end `infer.py` runs with the corresponding git commit.
Before remote benchmark runs, check GPU occupancy with `nvidia-smi`; if another
process is using the GPU, stop and do not run inference.

| Commit     |      AUC |     PCOC |  Latency | score_latency | score_model | score_all | Notes                                                 |
| ---------- | -------: | -------: | -------: | ------------: | ----------: | --------: | ----------------------------------------------------- |
| `4d9d8f50` | 0.760742 | 1.063921 | 17.1683s |      0.942772 |    0.323558 | 75.700802 | Remote A800, synced local source, CUDA Graph enabled. |
| `4d9d8f50` | 0.760742 | 1.063921 | 18.8907s |      0.937031 |    0.323558 | 75.298904 | Judge-style run: no env flags, no CLI flags.          |
| `24655cc`  | 0.760742 | 1.063921 | 22.3309s |      0.925564 |    0.323558 | 74.496183 | Corrected baseline timer; graph setup in `load_model()`. |

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
