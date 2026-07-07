from pathlib import Path
import argparse
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_LIBRARIES = (REPO_ROOT / "libraries").resolve()

if PROJECT_LIBRARIES.exists():
    sys.path.insert(0, str(PROJECT_LIBRARIES))

try:
    import torch
except ImportError as exc:
    raise SystemExit("Failed to import PyTorch from the active Python environment.") from exc


def percentile(sorted_values, pct):
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize(values, name):
    values = sorted(values)
    if not values:
        print(f"{name}: empty")
        return
    print(
        f"{name}: count={len(values)}, "
        f"min={values[0]}, "
        f"p50={percentile(values, 50):.2f}, "
        f"p90={percentile(values, 90):.2f}, "
        f"p95={percentile(values, 95):.2f}, "
        f"p99={percentile(values, 99):.2f}, "
        f"max={values[-1]}, "
        f"mean={statistics.fmean(values):.2f}"
    )


def load_batches(cache_dir, max_shards=None):
    shard_files = sorted(
        cache_dir.glob("shard_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files found under {cache_dir}")

    for shard_file in shard_files:
        shard_batches = torch.load(shard_file, weights_only=False, map_location="cpu")
        print(f"[INFO] loaded {len(shard_batches)} batches from {shard_file}")
        for batch in shard_batches:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "code" / "dataset" / "cached_batches",
        help="Directory containing shard_*.pt cached batch files.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit the number of shards to scan for quick checks.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    print(f"[INFO] cache_dir={cache_dir}")

    batch_seq_lens = []
    batch_user_counts = []
    user_lens = []
    dense_attention_scores = 0
    varlen_causal_scores = 0
    varlen_full_scores = 0
    num_batches = 0

    for batch in load_batches(cache_dir, args.max_shards):
        offsets = batch["user_offsets"].view(-1).to(torch.long)
        lengths = (offsets[1:] - offsets[:-1]).tolist()
        seq_len = int(offsets[-1].item())
        user_count = len(lengths)

        num_batches += 1
        batch_seq_lens.append(seq_len)
        batch_user_counts.append(user_count)
        user_lens.extend(int(length) for length in lengths)

        dense_attention_scores += seq_len * seq_len
        varlen_full_scores += sum(int(length) * int(length) for length in lengths)
        varlen_causal_scores += sum(int(length) * (int(length) + 1) // 2 for length in lengths)

    print(f"[INFO] scanned_batches={num_batches}")
    summarize(batch_seq_lens, "batch_seq_len")
    summarize(batch_user_counts, "batch_user_count")
    summarize(user_lens, "user_seq_len")

    if dense_attention_scores:
        print(f"dense_attention_scores=sum(S^2): {dense_attention_scores}")
        print(f"varlen_full_scores=sum(L^2): {varlen_full_scores}")
        print(f"varlen_causal_scores=sum(L*(L+1)/2): {varlen_causal_scores}")
        print(f"varlen_full / dense: {varlen_full_scores / dense_attention_scores:.6f}")
        print(f"varlen_causal / dense: {varlen_causal_scores / dense_attention_scores:.6f}")
        print(f"dense / varlen_causal: {dense_attention_scores / varlen_causal_scores:.2f}x")


if __name__ == "__main__":
    main()
