from pathlib import Path
import argparse
import csv
import datetime as _datetime
import os
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = REPO_ROOT / "code" / "test_cuda_attention.py"
DEFAULT_CLOUD_PYTHON = Path("/opt/conda/envs/python35-paddle120-env/bin/python")

CANDIDATES = (
    (8, 64, 128),
    (16, 64, 128),
    (32, 64, 128),
    (32, 64, 256),
    (16, 128, 256),
)

BENCH_RE = re.compile(
    r"\[BENCH\].*?"
    r"varlen_causal_attention=([0-9.]+) ms,\s*"
    r"varlen_causal_attention_tiled=([0-9.]+) ms,\s*"
    r"(?:varlen_causal_attention_tiled_compact=([0-9.]+) ms,\s*)?"
    r"(?:varlen_causal_attention_mma_qk=([0-9.]+) ms,\s*)?"
    r"(?:varlen_causal_attention_mma_qk_pv=([0-9.]+) ms,\s*)?"
    r"(?:varlen_causal_attention_mma_qk_pv_padded=([0-9.]+) ms,\s*)?"
    r"(?:varlen_causal_attention_mma_qk_pv_padded_shared_kv=([0-9.]+) ms,\s*)?"
    r"(?:varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta=([0-9.]+) ms,\s*)?"
    r"dense_torch=([0-9.]+) ms"
)


def default_python():
    if DEFAULT_CLOUD_PYTHON.exists():
        return str(DEFAULT_CLOUD_PYTHON)
    return sys.executable


def parse_candidates(raw_candidates):
    if not raw_candidates:
        return list(CANDIDATES)

    parsed = []
    for item in raw_candidates:
        parts = item.split(",")
        if len(parts) != 3:
            raise SystemExit(f"Invalid candidate {item!r}; expected BR,BC,THREADS")
        try:
            br, bc, threads = (int(part) for part in parts)
        except ValueError as exc:
            raise SystemExit(f"Invalid candidate {item!r}; all values must be integers") from exc
        parsed.append((br, bc, threads))
    return parsed


def run_and_tee(cmd, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_parts = []

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(
            "env ATTENTION_TILED_BR={br} ATTENTION_TILED_BC={bc} "
            "ATTENTION_TILED_THREADS={threads} TORCH_EXTENSIONS_DIR={ext_dir}\n\n".format(
                br=env.get("ATTENTION_TILED_BR", ""),
                bc=env.get("ATTENTION_TILED_BC", ""),
                threads=env.get("ATTENTION_TILED_THREADS", ""),
                ext_dir=env.get("TORCH_EXTENSIONS_DIR", ""),
            )
        )
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            output_parts.append(line)
        returncode = proc.wait()

    return returncode, "".join(output_parts)


def parse_bench_output(output):
    match = BENCH_RE.search(output)
    if not match:
        return None
    one_query_ms = float(match.group(1))
    tiled_ms = float(match.group(2))
    compact_ms = float(match.group(3)) if match.group(3) is not None else ""
    mma_qk_ms = float(match.group(4)) if match.group(4) is not None else ""
    mma_qk_pv_ms = float(match.group(5)) if match.group(5) is not None else ""
    mma_qk_pv_padded_ms = float(match.group(6)) if match.group(6) is not None else ""
    mma_qk_pv_padded_shared_kv_ms = float(match.group(7)) if match.group(7) is not None else ""
    mma_qk_pv_padded_shared_kv_meta_ms = float(match.group(8)) if match.group(8) is not None else ""
    dense_ms = float(match.group(9))
    return {
        "one_query_ms": one_query_ms,
        "tiled_ms": tiled_ms,
        "compact_ms": compact_ms,
        "mma_qk_ms": mma_qk_ms,
        "mma_qk_pv_ms": mma_qk_pv_ms,
        "mma_qk_pv_padded_ms": mma_qk_pv_padded_ms,
        "mma_qk_pv_padded_shared_kv_ms": mma_qk_pv_padded_shared_kv_ms,
        "mma_qk_pv_padded_shared_kv_meta_ms": mma_qk_pv_padded_shared_kv_meta_ms,
        "dense_torch_ms": dense_ms,
    }


def write_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "br",
        "bc",
        "threads",
        "bench_users",
        "bench_len",
        "iters",
        "returncode",
        "one_query_ms",
        "tiled_ms",
        "compact_ms",
        "mma_qk_ms",
        "mma_qk_pv_ms",
        "mma_qk_pv_padded_ms",
        "mma_qk_pv_padded_shared_kv_ms",
        "mma_qk_pv_padded_shared_kv_meta_ms",
        "dense_torch_ms",
        "log",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep tiled CUDA attention Br/Bc/thread compile-time parameters."
    )
    parser.add_argument("--python", default=default_python(), help="Python executable used to run test_cuda_attention.py")
    parser.add_argument("--bench-lens", type=int, nargs="+", default=[226, 401])
    parser.add_argument("--bench-users", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Override/add candidate as BR,BC,THREADS. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for logs and summary.csv. Default: code/attention_sweep_runs/<timestamp>",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "code" / "attention_sweep_runs" / timestamp
    elif not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    candidates = parse_candidates(args.candidate)
    rows = []

    print(f"[INFO] python: {args.python}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] candidates: {candidates}")
    print(f"[INFO] bench_lens: {args.bench_lens}, bench_users={args.bench_users}, iters={args.iters}")

    for br, bc, threads in candidates:
        config_name = f"br{br}_bc{bc}_th{threads}"
        ext_dir = output_dir / "torch_extensions" / config_name

        for bench_len in args.bench_lens:
            log_path = output_dir / f"{config_name}_len{bench_len}.log"
            cmd = [
                args.python,
                str(TEST_SCRIPT),
                "--bench-users",
                str(args.bench_users),
                "--bench-len",
                str(bench_len),
                "--iters",
                str(args.iters),
            ]
            env = os.environ.copy()
            env["ATTENTION_TILED_BR"] = str(br)
            env["ATTENTION_TILED_BC"] = str(bc)
            env["ATTENTION_TILED_THREADS"] = str(threads)
            env["TORCH_EXTENSIONS_DIR"] = str(ext_dir)

            print("\n" + "=" * 96)
            print(f"[RUN] {config_name}, bench_len={bench_len}")
            print(f"[LOG] {log_path}")
            print("[CMD] " + " ".join(cmd))

            row = {
                "br": br,
                "bc": bc,
                "threads": threads,
                "bench_users": args.bench_users,
                "bench_len": bench_len,
                "iters": args.iters,
                "returncode": "",
                "one_query_ms": "",
                "tiled_ms": "",
                "compact_ms": "",
                "mma_qk_ms": "",
                "mma_qk_pv_ms": "",
                "mma_qk_pv_padded_ms": "",
                "mma_qk_pv_padded_shared_kv_ms": "",
                "mma_qk_pv_padded_shared_kv_meta_ms": "",
                "dense_torch_ms": "",
                "log": str(log_path),
            }

            if args.dry_run:
                row["returncode"] = "dry-run"
                rows.append(row)
                continue

            returncode, output = run_and_tee(cmd, env, log_path)
            row["returncode"] = returncode
            parsed = parse_bench_output(output)
            if parsed is not None:
                row.update(parsed)
                print(
                    "[RESULT] {config}, len={bench_len}, one_query={one:.6f} ms, "
                    "tiled={tiled:.6f} ms, compact={compact}, mma_qk={mma_qk}, "
                    "mma_qk_pv={mma_qk_pv}, mma_qk_pv_padded={mma_qk_pv_padded}, "
                    "mma_qk_pv_padded_shared_kv={mma_qk_pv_padded_shared_kv}, "
                    "mma_qk_pv_padded_shared_kv_meta={mma_qk_pv_padded_shared_kv_meta}, "
                    "dense={dense:.6f} ms".format(
                        config=config_name,
                        bench_len=bench_len,
                        one=parsed["one_query_ms"],
                        tiled=parsed["tiled_ms"],
                        compact=(
                            f"{parsed['compact_ms']:.6f} ms"
                            if parsed["compact_ms"] != ""
                            else "n/a"
                        ),
                        mma_qk=(
                            f"{parsed['mma_qk_ms']:.6f} ms"
                            if parsed["mma_qk_ms"] != ""
                            else "n/a"
                        ),
                        mma_qk_pv=(
                            f"{parsed['mma_qk_pv_ms']:.6f} ms"
                            if parsed["mma_qk_pv_ms"] != ""
                            else "n/a"
                        ),
                        mma_qk_pv_padded=(
                            f"{parsed['mma_qk_pv_padded_ms']:.6f} ms"
                            if parsed["mma_qk_pv_padded_ms"] != ""
                            else "n/a"
                        ),
                        mma_qk_pv_padded_shared_kv=(
                            f"{parsed['mma_qk_pv_padded_shared_kv_ms']:.6f} ms"
                            if parsed["mma_qk_pv_padded_shared_kv_ms"] != ""
                            else "n/a"
                        ),
                        mma_qk_pv_padded_shared_kv_meta=(
                            f"{parsed['mma_qk_pv_padded_shared_kv_meta_ms']:.6f} ms"
                            if parsed["mma_qk_pv_padded_shared_kv_meta_ms"] != ""
                            else "n/a"
                        ),
                        dense=parsed["dense_torch_ms"],
                    )
                )
            else:
                print(f"[WARNING] No [BENCH] line parsed for {config_name}, bench_len={bench_len}")

            rows.append(row)
            write_summary(output_dir / "summary.csv", rows)

            if returncode != 0 and args.fail_fast:
                raise SystemExit(returncode)

    write_summary(output_dir / "summary.csv", rows)
    print("\n[INFO] sweep complete")
    print(f"[INFO] summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
