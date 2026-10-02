from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import psutil

from cs336_basics.train_bpe import merge as train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="May be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    return parser.parse_args()


def peak_rss_bytes() -> int:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return peak_rss

    return peak_rss * 1024


def save_pickle(path: Path, value: object) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> object:
    with path.open("rb") as file:
        return pickle.load(file)


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    memory_before = process.memory_info().rss
    start_time = time.perf_counter()

    print(f"Dataset: {args.dataset_name}")
    print(f"Input: {args.input}")
    print(f"Input size: {args.input.stat().st_size:,} bytes")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"Special tokens: {args.special_token}")
    print("Starting BPE training...", flush=True)

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
    )

    elapsed_seconds = time.perf_counter() - start_time
    memory_after = process.memory_info().rss
    peak_memory = peak_rss_bytes()

    if len(vocab) != args.vocab_size:
        raise RuntimeError(
            f"Expected vocab size {args.vocab_size}, got {len(vocab)}"
        )

    longest_token = max(vocab.values(), key=len)

    try:
        longest_token_text = longest_token.decode("utf-8")
    except UnicodeDecodeError:
        longest_token_text = None

    vocab_path = args.output_dir / "vocab.pkl"
    merges_path = args.output_dir / "merges.pkl"
    metrics_path = args.output_dir / "metrics.json"

    save_pickle(vocab_path, vocab)
    save_pickle(merges_path, merges)

    metrics = {
        "dataset_name": args.dataset_name,
        "input_path": str(args.input.resolve()),
        "input_size_bytes": args.input.stat().st_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "elapsed_seconds": elapsed_seconds,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "peak_rss_bytes": peak_memory,
        "longest_token_hex": longest_token.hex(),
        "longest_token_num_bytes": len(longest_token),
        "longest_token_text": longest_token_text,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Verify that serialization is lossless.
    loaded_vocab = load_pickle(vocab_path)
    loaded_merges = load_pickle(merges_path)

    if loaded_vocab != vocab:
        raise RuntimeError("Reloaded vocab does not match original vocab")

    if loaded_merges != merges:
        raise RuntimeError("Reloaded merges do not match original merges")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()