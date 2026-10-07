from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


UINT16_MAX = np.iinfo(np.uint16).max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-tokenize train and validation text files into uint16 NumPy arrays."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="May be passed multiple times.",
    )
    parser.add_argument(
        "--buffer-tokens",
        type=int,
        default=1_000_000,
        help="Number of token IDs buffered before writing to disk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output arrays.",
    )
    return parser.parse_args()


def flush_token_buffer(buffer: list[int], output_file) -> int:
    if not buffer:
        return 0

    minimum = min(buffer)
    maximum = max(buffer)
    if minimum < 0 or maximum > UINT16_MAX:
        raise ValueError(
            f"Token IDs must fit in uint16, but this buffer contains IDs in [{minimum}, {maximum}]."
        )

    token_array = np.asarray(buffer, dtype=np.uint16)
    token_array.tofile(output_file)
    written = len(buffer)
    buffer.clear()
    return written


def raw_tokens_to_npy(
    raw_path: Path,
    output_path: Path,
    token_count: int,
    copy_tokens: int,
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp.npy",
        delete=False,
    ) as temporary_file:
        temporary_npy_path = Path(temporary_file.name)

    try:
        destination = np.lib.format.open_memmap(
            temporary_npy_path,
            mode="w+",
            dtype=np.uint16,
            shape=(token_count,),
        )

        if token_count:
            source = np.memmap(
                raw_path,
                mode="r",
                dtype=np.uint16,
                shape=(token_count,),
            )
            for start in range(0, token_count, copy_tokens):
                end = min(start + copy_tokens, token_count)
                destination[start:end] = source[start:end]
            del source

        destination.flush()
        del destination
        os.replace(temporary_npy_path, output_path)
    finally:
        temporary_npy_path.unlink(missing_ok=True)


def encode_file(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    buffer_tokens: int,
    overwrite: bool,
) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_size = input_path.stat().st_size
    started = time.perf_counter()
    token_count = 0
    buffer: list[int] = []

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tokens.bin",
        delete=False,
    ) as raw_file:
        raw_path = Path(raw_file.name)

        try:
            with input_path.open("r", encoding="utf-8") as input_file:
                for token_id in tokenizer.encode_iterable(input_file):
                    buffer.append(token_id)
                    if len(buffer) >= buffer_tokens:
                        token_count += flush_token_buffer(buffer, raw_file)
                        elapsed = time.perf_counter() - started
                        print(
                            f"{input_path.name}: {token_count:,} tokens "
                            f"({token_count / max(elapsed, 1e-9):,.0f} tokens/s)",
                            flush=True,
                        )

            token_count += flush_token_buffer(buffer, raw_file)
            raw_file.flush()
            os.fsync(raw_file.fileno())

            raw_tokens_to_npy(
                raw_path=raw_path,
                output_path=output_path,
                token_count=token_count,
                copy_tokens=buffer_tokens,
            )
        finally:
            raw_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    saved = np.load(output_path, mmap_mode="r")
    if saved.dtype != np.uint16 or saved.shape != (token_count,):
        raise RuntimeError(
            f"Invalid saved array: dtype={saved.dtype}, shape={saved.shape}, expected ({token_count},)."
        )
    del saved

    metrics: dict[str, object] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_size_bytes": input_size,
        "token_count": token_count,
        "dtype": "uint16",
        "token_data_size_bytes": token_count * np.dtype(np.uint16).itemsize,
        "elapsed_seconds": elapsed,
        "throughput_bytes_per_second": input_size / elapsed if elapsed else None,
        "compression_ratio_bytes_per_token": input_size / token_count if token_count else None,
    }
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    if args.buffer_tokens <= 0:
        raise ValueError("--buffer-tokens must be positive.")

    vocab_path = args.tokenizer_dir / "vocab.pkl"
    merges_path = args.tokenizer_dir / "merges.pkl"
    if not vocab_path.is_file():
        raise FileNotFoundError(vocab_path)
    if not merges_path.is_file():
        raise FileNotFoundError(merges_path)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=args.special_token,
    )

    if len(tokenizer.vocab) > UINT16_MAX + 1:
        raise ValueError(f"Vocabulary of size {len(tokenizer.vocab):,} does not fit in uint16.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output = args.output_dir / f"{args.dataset_name}_train.npy"
    valid_output = args.output_dir / f"{args.dataset_name}_valid.npy"

    metrics = {
        "dataset_name": args.dataset_name,
        "tokenizer_dir": str(args.tokenizer_dir),
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": args.special_token,
        "train": encode_file(
            tokenizer=tokenizer,
            input_path=args.train_input,
            output_path=train_output,
            buffer_tokens=args.buffer_tokens,
            overwrite=args.overwrite,
        ),
        "valid": encode_file(
            tokenizer=tokenizer,
            input_path=args.valid_input,
            output_path=valid_output,
            buffer_tokens=args.buffer_tokens,
            overwrite=args.overwrite,
        ),
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
