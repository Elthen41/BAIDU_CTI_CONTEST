# Infer Score Ledger

This file records end-to-end `infer.py` runs with the corresponding git commit.
Before remote benchmark runs, check GPU occupancy with `nvidia-smi`; if another
process is using the GPU, stop and do not run inference.

| Commit     |      AUC |     PCOC |  Latency | score_latency | score_model | score_all | Notes                                                 |
| ---------- | -------: | -------: | -------: | ------------: | ----------: | --------: | ----------------------------------------------------- |
| `4d9d8f50` | 0.760742 | 1.063921 | 17.1683s |      0.942772 |    0.323558 | 75.700802 | Remote A800, synced local source, CUDA Graph enabled. |
| `4d9d8f50` | 0.760742 | 1.063921 | 18.8907s |      0.937031 |    0.323558 | 75.298904 | Judge-style run: no env flags, no CLI flags.          |
| `24655cc`  | 0.760742 | 1.063921 | 22.3309s |      0.925564 |    0.323558 | 74.496183 | Corrected baseline timer; graph setup in `load_model()`. |
| 7b0341a      | 0.761060 | 1.063562 | 13.7868s |      0.954044 |    0.324508 | 76.518301 | Add prepin |
| 3527b6c | 0.761060 | 1.063562 | 13.4897s |      0.946200 |    0.3245086. | 76.587628 | `USE_JUDGE_MOVE_PREFETCH=1` |

## Run Notes

### CUDA Graph static prefetch attempts

- Double-buffer graph prefetch was attempted by capturing a second graph/static buffer per bucket. It was interrupted before a valid score because setup took several minutes and GPU memory reached about `49.5GB`.
- Lightweight single-buffer graph prefetch was then attempted by scheduling next-batch copies into graph static buffers after replay and returning separate `logid` / `pred_mask` tensors for output postprocessing. It avoided the memory increase but still failed to reach a usable timed loop in the remote run.
- Current source keeps this behind `USE_CUDA_GRAPH_STATIC_PREFETCH=1`. `USE_JUDGE_MOVE_PREFETCH=1` alone should not enable graph-static prefetch.
