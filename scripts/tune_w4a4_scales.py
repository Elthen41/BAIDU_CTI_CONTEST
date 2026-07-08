#!/usr/bin/env python3
"""Grid-search SMoE W4A4 scale environment variables.

The script repeatedly runs infer.py with USE_SIMPLE_W4A4_FC1_SMOE=1 and records
AUC/PCOC/latency from stdout. It is intended for remote tuning, where labels and
the prebuilt CUDA extension are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan"
PATTERNS = {
    "inference_time": re.compile(r"inference time:\s*(" + FLOAT_RE + r")s", re.I),
    "auc": re.compile(r"\bAUC:\s*(" + FLOAT_RE + r")", re.I),
    "pcoc": re.compile(r"\bPCOC:\s*(" + FLOAT_RE + r")", re.I),
    "latency": re.compile(r"\bLatency:\s*(" + FLOAT_RE + r")s", re.I),
    "score_latency": re.compile(r"\bscore_latency:\s*(" + FLOAT_RE + r")", re.I),
    "score_model": re.compile(r"\bscore_model:\s*(" + FLOAT_RE + r")", re.I),
    "score_all": re.compile(r"\bscore_all:\s*(" + FLOAT_RE + r")", re.I),
}

TUNED_KEYS = (
    "SIMPLE_W4A4_FC1_ACT_SCALE",
    "SIMPLE_W4A4_ACT_SCALE",
    "SIMPLE_W4A4_FC1_OUTPUT_SCALE",
    "SIMPLE_W4A4_FC2_OUTPUT_SCALE",
    "SIMPLE_W4A4_WEIGHT_SCALE",
)

DEFAULT_AUC_MIN = 0.75
DEFAULT_PCOC_MIN = 0.85
DEFAULT_PCOC_MAX = 1.15


def parse_float_list(text: str, *, allow_zero: bool = False) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0 or (value == 0.0 and not allow_zero):
            raise argparse.ArgumentTypeError(f"scale values must be positive, got {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_env_pair(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"empty env key in {text!r}")
    return key, value


def fmt_float(value: float) -> str:
    return f"{value:.8g}"


def parse_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, pattern in PATTERNS.items():
        match = None
        for match in pattern.finditer(output):
            pass
        if match is not None:
            raw = match.group(1).lower()
            metrics[name] = float(raw)
    return metrics


def score_valid(
    row: dict[str, object],
    *,
    auc_min: float = DEFAULT_AUC_MIN,
    pcoc_min: float = DEFAULT_PCOC_MIN,
    pcoc_max: float = DEFAULT_PCOC_MAX,
) -> bool:
    auc = row.get("auc")
    pcoc = row.get("pcoc")
    return (
        isinstance(auc, float)
        and isinstance(pcoc, float)
        and auc_min <= auc <= 1.0
        and pcoc_min <= pcoc <= pcoc_max
    )


def trial_key(env_updates: dict[str, str]) -> str:
    material = json.dumps({k: env_updates.get(k, "") for k in TUNED_KEYS}, sort_keys=True)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def load_completed(jsonl_path: Path) -> set[str]:
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("trial_key")
            if isinstance(key, str):
                completed.add(key)
    return completed


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rewrite_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    auc_min: float = DEFAULT_AUC_MIN,
    pcoc_min: float = DEFAULT_PCOC_MIN,
    pcoc_max: float = DEFAULT_PCOC_MAX,
) -> None:
    fieldnames = [
        "rank",
        "valid",
        "score_all",
        "score_model",
        "auc",
        "pcoc",
        "latency",
        "inference_time",
        "returncode",
        "elapsed_sec",
        "trial_key",
        *TUNED_KEYS,
        "log_path",
    ]
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            0 if score_valid(r, auc_min=auc_min, pcoc_min=pcoc_min, pcoc_max=pcoc_max) else 1,
            -float(r.get("score_all", -1.0) or -1.0),
            abs(float(r.get("pcoc", 999.0) or 999.0) - 1.0),
            -float(r.get("auc", -1.0) or -1.0),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            out = {name: row.get(name, "") for name in fieldnames}
            out["rank"] = rank
            out["valid"] = int(score_valid(row, auc_min=auc_min, pcoc_min=pcoc_min, pcoc_max=pcoc_max))
            writer.writerow(out)


def load_rows(jsonl_path: Path) -> list[dict[str, object]]:
    rows = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def make_trials(args: argparse.Namespace) -> list[dict[str, str]]:
    weight_values = args.weight_scale
    trials = []
    for fc1_act, fc2_act, fc1_out, fc2_out, weight in itertools.product(
        args.fc1_act,
        args.fc2_act,
        args.fc1_out,
        args.fc2_out,
        weight_values,
    ):
        env = {
            "USE_SIMPLE_W4A4_SMOE": "1",
            "USE_SIMPLE_W4A4_FC1_SMOE": "1",
            "USE_SIMPLE_W4A4_FC1_PACK_OUTPUT": "1" if args.pack_output else "0",
            "SIMPLE_W4A4_FC1_ACT_SCALE": fmt_float(fc1_act),
            "SIMPLE_W4A4_ACT_SCALE": fmt_float(fc2_act),
            "SIMPLE_W4A4_FC1_OUTPUT_SCALE": fmt_float(fc1_out),
            "SIMPLE_W4A4_FC2_OUTPUT_SCALE": fmt_float(fc2_out),
        }
        if weight > 0.0:
            env["SIMPLE_W4A4_WEIGHT_SCALE"] = fmt_float(weight)
        else:
            env["SIMPLE_W4A4_WEIGHT_SCALE"] = "0"
        for key, value in args.extra_env:
            env[key] = value
        trials.append(env)

    if args.shuffle:
        random.Random(args.seed).shuffle(trials)
    if args.limit is not None:
        trials = trials[: args.limit]
    return trials


def run_trial(
    args: argparse.Namespace,
    env_updates: dict[str, str],
    out_dir: Path,
    index: int,
    total: int,
) -> dict[str, object]:
    key = trial_key(env_updates)
    log_path = out_dir / "logs" / f"{index:04d}_{key}.log"
    env = os.environ.copy()
    env.update(env_updates)

    cmd = [args.python, str(args.infer)]
    if args.ckpt:
        cmd.extend(["--ckpt", str(args.ckpt)])

    started = time.time()
    print(
        f"[{index}/{total}] {key} "
        f"fc1_act={env_updates['SIMPLE_W4A4_FC1_ACT_SCALE']} "
        f"fc2_act={env_updates['SIMPLE_W4A4_ACT_SCALE']} "
        f"fc1_out={env_updates['SIMPLE_W4A4_FC1_OUTPUT_SCALE']} "
        f"fc2_out={env_updates['SIMPLE_W4A4_FC2_OUTPUT_SCALE']} "
        f"wscale={env_updates['SIMPLE_W4A4_WEIGHT_SCALE']}",
        flush=True,
    )

    if args.dry_run:
        return {
            "trial_key": key,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "log_path": str(log_path),
            **env_updates,
        }

    proc = subprocess.run(
        cmd,
        cwd=str(args.cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    elapsed = time.time() - started
    log_path.write_text(proc.stdout, encoding="utf-8")

    row: dict[str, object] = {
        "trial_key": key,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 4),
        "log_path": str(log_path),
        **env_updates,
    }
    row.update(parse_metrics(proc.stdout))
    if proc.returncode != 0:
        row["error_tail"] = "\n".join(proc.stdout.splitlines()[-40:])
    return row


def print_top(
    rows: list[dict[str, object]],
    n: int,
    *,
    auc_min: float = DEFAULT_AUC_MIN,
    pcoc_min: float = DEFAULT_PCOC_MIN,
    pcoc_max: float = DEFAULT_PCOC_MAX,
) -> None:
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if score_valid(r, auc_min=auc_min, pcoc_min=pcoc_min, pcoc_max=pcoc_max) else 1,
            -float(r.get("score_all", -1.0) or -1.0),
            abs(float(r.get("pcoc", 999.0) or 999.0) - 1.0),
            -float(r.get("auc", -1.0) or -1.0),
        ),
    )
    print("\nTop candidates:")
    for i, row in enumerate(ranked[:n], start=1):
        print(
            f"{i:2d}. valid={int(score_valid(row, auc_min=auc_min, pcoc_min=pcoc_min, pcoc_max=pcoc_max))} "
            f"score_all={row.get('score_all', '')} "
            f"auc={row.get('auc', '')} pcoc={row.get('pcoc', '')} "
            f"lat={row.get('latency', row.get('inference_time', ''))} "
            f"fc1_act={row.get('SIMPLE_W4A4_FC1_ACT_SCALE', '')} "
            f"fc2_act={row.get('SIMPLE_W4A4_ACT_SCALE', '')} "
            f"fc1_out={row.get('SIMPLE_W4A4_FC1_OUTPUT_SCALE', '')} "
            f"fc2_out={row.get('SIMPLE_W4A4_FC2_OUTPUT_SCALE', '')} "
            f"wscale={row.get('SIMPLE_W4A4_WEIGHT_SCALE', '')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", type=Path, default=Path("infer.py"))
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("scale_tuning/w4a4_fc1"))
    parser.add_argument("--fc1-act", type=parse_float_list, default=parse_float_list("0.5,1,2"))
    parser.add_argument("--fc2-act", type=parse_float_list, default=parse_float_list("0.25,0.5,1,2"))
    parser.add_argument("--fc1-out", type=parse_float_list, default=parse_float_list("0.75,1,1.25"))
    parser.add_argument("--fc2-out", type=parse_float_list, default=parse_float_list("0.9,1,1.1,1.25"))
    parser.add_argument(
        "--weight-scale",
        type=lambda s: parse_float_list(s, allow_zero=True),
        default=parse_float_list("0", allow_zero=True),
        help="0 means use infer.py auto scale. Positive values override SIMPLE_W4A4_WEIGHT_SCALE.",
    )
    parser.add_argument("--extra-env", type=parse_env_pair, action="append", default=[])
    parser.add_argument("--pack-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--auc-min", type=float, default=DEFAULT_AUC_MIN)
    parser.add_argument("--pcoc-min", type=float, default=DEFAULT_PCOC_MIN)
    parser.add_argument("--pcoc-max", type=float, default=DEFAULT_PCOC_MAX)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.cwd = args.cwd.resolve()
    args.infer = (args.cwd / args.infer).resolve() if not args.infer.is_absolute() else args.infer
    args.out_dir = (args.cwd / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)

    jsonl_path = args.out_dir / "results.jsonl"
    csv_path = args.out_dir / "results.csv"

    trials = make_trials(args)
    completed = load_completed(jsonl_path) if args.resume else set()
    rows = load_rows(jsonl_path) if args.resume else []

    pending = [env for env in trials if trial_key(env) not in completed]
    print(
        f"total={len(trials)} completed={len(completed)} pending={len(pending)} "
        f"auc_min={args.auc_min} pcoc=[{args.pcoc_min},{args.pcoc_max}] "
        f"out={args.out_dir}"
    )

    for done, env_updates in enumerate(pending, start=1):
        row = run_trial(args, env_updates, args.out_dir, done, len(pending))
        append_jsonl(jsonl_path, row)
        rows.append(row)
        rewrite_csv(
            csv_path,
            rows,
            auc_min=args.auc_min,
            pcoc_min=args.pcoc_min,
            pcoc_max=args.pcoc_max,
        )
        if row.get("returncode") == 0:
            print(
                f"    -> auc={row.get('auc', '')} pcoc={row.get('pcoc', '')} "
                f"score_all={row.get('score_all', '')} latency={row.get('latency', row.get('inference_time', ''))}",
                flush=True,
            )
        else:
            print(f"    -> failed returncode={row.get('returncode')} log={row.get('log_path')}", flush=True)

    rewrite_csv(
        csv_path,
        rows,
        auc_min=args.auc_min,
        pcoc_min=args.pcoc_min,
        pcoc_max=args.pcoc_max,
    )
    print_top(rows, args.top, auc_min=args.auc_min, pcoc_min=args.pcoc_min, pcoc_max=args.pcoc_max)
    print(f"\nWrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
