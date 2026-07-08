# Repository Guidelines

## Contest Context

This repository is for the BAIDU_CTI Contest CTR inference optimization task.
The scoring path is `infer.py`; optimize inference latency while preserving AUC
and PCOC constraints described in `guide.md`.

The official reference implementation is `infer_baseline.py`. Use it as the
behavioral contract for data loading, model structure, checkpoint loading, output
format, and metric calculation.

## Judge Behavior

- The submitted archive must include at least `infer.py`, `build_env.sh`, and
  `requirements.txt`.
- The judge first runs `build_env.sh`, then runs `infer.py`.
- The judge uses a fixed `main()` equivalent to `infer_baseline.py`.
- Treat changes to `main()` in local `infer.py` as debug-only; they will not
  affect judging.
- In judge execution, only `--ckpt` should be considered a reliable CLI
  argument. Other current `infer.py` CLI flags are local diagnostics only.
- Do not rely on local paths such as `code/dataset` for judged behavior unless
  the fixed baseline-compatible `main()` can reach the same files.

## Optimization Targets

- Keep `infer_baseline.py` as the official baseline; do not edit it unless the
  user explicitly asks.
- Keep correctness gates in mind: PCOC must stay in `[0.85, 1.15]` and AUC must
  stay in `[0.65, 1]`, otherwise the final score is zero.
- Default optimization work should focus on import-time setup, `load_model()`,
  model forward paths, CUDA extensions, and build artifacts that survive the
  fixed judge `main()`.
- Prefer default-on optimizations with environment-variable emergency fallbacks
  for risky CUDA paths.

## CUDA Build Direction

The judge build path uses CMake from `build_env.sh` and installs prebuilt Python
extension modules into `cmake_extensions`.

`infer.py` should import those prebuilt modules at runtime. The old
`torch.utils.cpp_extension.load` path is retained only as a local debugging
fallback when prebuilt modules are unavailable.

When changing the CMake path:

- Keep `build_env.sh` as the single judge build entry point.
- Ensure compiled Python extension modules are importable from the repository
  root or `libraries` without relying on manual shell setup.
- Preserve fallback behavior for local debugging until the CMake path is proven
  equivalent.
- Avoid introducing generated build outputs into the submission unless they are
  required by the judge and allowed by `guide.md`.

## Local Notes

- Remote server notes below refer to the team's A800 debug environment only.
  That server is useful for profiling and optimization before submission, but it
  is not the judge system and should not be treated as the source of judge paths
  or CLI behavior.
- Treat the remote access details in this file as private local workflow notes.
  Do not copy them into public docs, source comments, submissions, or issue/PR
  text.
- Use `rg` for repository searches.
- Before packaging, verify that caches, datasets, checkpoints not required for
  submission, and transient build outputs are excluded.

## Remote A800 Debug Server

Use two terminals.

Terminal 1 starts the FRP server:

```bash
ssh ubuntu@101.42.65.198
# passwd: vps114514baiducti666!?AXY
cd /home/ubuntu/frp/frp_0.65.0_linux_amd64
sudo ./frps -c frps.toml
```

After terminal 1 is running, terminal 2 connects through the exposed mini SSH
service:

```bash
ssh ubuntu@101.42.65.198
# passwd: vps114514baiducti666!?AXY
ssh -p 60022 aistudio@101.42.65.198
# passwd: 你的mini_sshd密码
```
