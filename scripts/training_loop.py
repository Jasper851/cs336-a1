"""Train the assignment Transformer language model on tokenized NumPy datasets.

The input files are expected to be one-dimensional ``.npy`` arrays containing
integer token IDs, such as the files produced by ``scripts/tokenize_datasets.py``.
They are opened with NumPy memory mapping, so the complete datasets do not need
to fit in RAM.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import Transformer_lm
from cs336_basics.train import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Define all dataset, model, optimizer, and logging configuration."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model on memory-mapped token arrays."
    )

    # Dataset and output paths.
    parser.add_argument("--train-data", type=Path, required=True, help="Tokenized training .npy file.")
    parser.add_argument("--valid-data", type=Path, required=True, help="Tokenized validation .npy file.")
    parser.add_argument("--experiment-name", required=True, help="Unique directory name for this run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Root directory containing all experiment runs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint from which to restore model, optimizer, and iteration.",
    )

    # Transformer hyperparameters.
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="SwiGLU hidden width. If omitted, the model computes its default width.",
    )
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # Optimization and learning-rate schedule hyperparameters.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="Iteration at which cosine decay reaches the minimum; defaults to max-iterations.",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Maximum gradient L2 norm. Set to 0 to disable clipping.",
    )

    # Evaluation, logging, checkpointing, and reproducibility controls.
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu, cuda, or mps. 'auto' selects an available accelerator.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on configurations that would otherwise cause unclear runtime errors."""
    positive_integer_fields = (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "batch_size",
        "max_iterations",
        "validation_interval",
        "validation_batches",
        "log_interval",
        "checkpoint_interval",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")

    if args.d_ff is not None and args.d_ff <= 0:
        raise ValueError("--d-ff must be positive when provided.")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters cannot be negative.")
    if args.min_learning_rate < 0 or args.max_learning_rate <= 0:
        raise ValueError("Learning rates must be non-negative, and the maximum must be positive.")
    if args.min_learning_rate > args.max_learning_rate:
        raise ValueError("--min-learning-rate cannot exceed --max-learning-rate.")
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        raise ValueError("AdamW beta values must lie in [0, 1).")
    if args.adam_eps <= 0 or args.weight_decay < 0 or args.gradient_clip < 0:
        raise ValueError("Adam epsilon must be positive; weight decay and gradient clipping cannot be negative.")

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iterations
    if args.cosine_cycle_iters <= args.warmup_iters:
        raise ValueError("--cosine-cycle-iters must be greater than --warmup-iters.")

    for path in (args.train_data, args.valid_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.resume_from is not None and not args.resume_from.is_file():
        raise FileNotFoundError(args.resume_from)
    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError("--experiment-name must be a single directory name.")


def choose_device(requested_device: str) -> torch.device:
    """Resolve the convenient 'auto' option while preserving explicit user choices."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_token_array(path: Path, context_length: int) -> np.memmap:
    """Open a tokenized ``.npy`` file without reading the full array into memory."""
    dataset = np.load(path, mmap_mode="r")
    if not isinstance(dataset, np.memmap):
        raise TypeError(f"Expected a memory-mapped .npy array at {path}.")
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError(f"Expected a one-dimensional integer token array, got shape={dataset.shape}, dtype={dataset.dtype}.")
    if len(dataset) <= context_length:
        raise ValueError(
            f"Dataset {path} contains {len(dataset)} tokens, but more than {context_length} are required."
        )
    return dataset


def write_metrics(log_path: Path | None, metrics: dict[str, object]) -> None:
    """Print one JSON record and optionally append the same record to a JSONL file."""
    record = json.dumps(metrics, sort_keys=True)
    print(record, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(record + "\n")


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    validation_batches: int,
    device: torch.device,
) -> float:
    """Estimate validation loss from several independently sampled batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(validation_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        total_loss += cross_entropy(logits, targets).item()
    model.train()
    return total_loss / validation_batches


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Apply the scheduled learning rate to every optimizer parameter group."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def main() -> None:
    args = parse_args()
    validate_args(args)

    experiment_dir = args.output_dir / args.experiment_name
    checkpoint_path = experiment_dir / "checkpoint.pt"
    log_path = experiment_dir / "metrics.jsonl"
    if log_path.exists() and args.resume_from is None:
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}. Choose a new name or pass --resume-from."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Seed all random-number generators used by this script. PyTorch controls
    # initialization; NumPy controls the random starting offsets in get_batch.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    train_data = load_token_array(args.train_data, args.context_length)
    valid_data = load_token_array(args.valid_data, args.context_length)

    # Passing the device into the constructor creates parameters directly on the
    # target device and avoids allocating a second full copy during model.to(...).
    model = Transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # A stored iteration is the number of already-completed optimizer updates.
    # Therefore range(start_iteration, ...) resumes at exactly the next update.
    start_iteration = 0
    if args.resume_from is not None:
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        if start_iteration > args.max_iterations:
            raise ValueError(
                f"Checkpoint has completed {start_iteration} iterations, exceeding "
                f"--max-iterations={args.max_iterations}."
            )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    write_metrics(
        log_path,
        {
            "event": "start",
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "device": str(device),
            "parameter_count": parameter_count,
            "train_tokens": len(train_data),
            "valid_tokens": len(valid_data),
            "start_iteration": start_iteration,
            "max_iterations": args.max_iterations,
        },
    )

    model.train()
    training_started = time.perf_counter()
    interval_started = training_started
    steps_since_log = 0
    completed_iterations = start_iteration
    last_checkpoint_iteration: int | None = None

    try:
        for iteration in range(start_iteration, args.max_iterations):
            # Schedule the learning rate before this iteration's optimizer step.
            learning_rate = get_lr_cosine_schedule(
                iteration,
                args.max_learning_rate,
                args.min_learning_rate,
                args.warmup_iters,
                args.cosine_cycle_iters,
            )
            set_optimizer_learning_rate(optimizer, learning_rate)

            inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

            # set_to_none avoids filling every gradient tensor with zeros and is
            # generally more memory-efficient than the default zeroing behavior.
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()

            if args.gradient_clip > 0:
                gradient_clipping(model.parameters(), args.gradient_clip)
            optimizer.step()
            completed_iterations = iteration + 1
            steps_since_log += 1

            # Training metrics use the current sampled batch. Throughput is
            # averaged only over the current logging interval.
            if completed_iterations % args.log_interval == 0 or completed_iterations == args.max_iterations:
                now = time.perf_counter()
                tokens_processed = steps_since_log * args.batch_size * args.context_length
                write_metrics(
                    log_path,
                    {
                        "event": "train",
                        "iteration": completed_iterations,
                        "loss": loss.item(),
                        "learning_rate": learning_rate,
                        "tokens_per_second": tokens_processed / max(now - interval_started, 1e-12),
                        "elapsed_seconds": now - training_started,
                    },
                )
                interval_started = now
                steps_since_log = 0

            if (
                completed_iterations % args.validation_interval == 0
                or completed_iterations == args.max_iterations
            ):
                validation_started = time.perf_counter()
                validation_loss = evaluate(
                    model,
                    valid_data,
                    args.batch_size,
                    args.context_length,
                    args.validation_batches,
                    device,
                )
                validation_finished = time.perf_counter()
                write_metrics(
                    log_path,
                    {
                        "event": "validation",
                        "iteration": completed_iterations,
                        "loss": validation_loss,
                        "elapsed_seconds": validation_finished - training_started,
                    },
                )
                # Shift the throughput timer forward so validation work is not
                # incorrectly counted as time spent processing training tokens.
                interval_started += time.perf_counter() - validation_started

            if completed_iterations % args.checkpoint_interval == 0:
                checkpoint_started = time.perf_counter()
                save_checkpoint(model, optimizer, completed_iterations, checkpoint_path)
                last_checkpoint_iteration = completed_iterations
                write_metrics(
                    log_path,
                    {
                        "event": "checkpoint",
                        "iteration": completed_iterations,
                        "path": str(checkpoint_path),
                    },
                )
                # Checkpoint I/O is likewise excluded from training throughput.
                interval_started += time.perf_counter() - checkpoint_started

    except KeyboardInterrupt:
        # Preserve all completed optimizer updates if the user intentionally
        # interrupts a long run. The interrupted, unfinished step is never saved.
        save_checkpoint(model, optimizer, completed_iterations, checkpoint_path)
        write_metrics(
            log_path,
            {
                "event": "interrupted",
                "iteration": completed_iterations,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        raise

    # Always leave a final resumable state, even when max-iterations is not a
    # multiple of checkpoint-interval.
    if last_checkpoint_iteration != completed_iterations:
        save_checkpoint(model, optimizer, completed_iterations, checkpoint_path)
    write_metrics(
        log_path,
        {
            "event": "complete",
            "iteration": completed_iterations,
            "checkpoint_path": str(checkpoint_path),
            "elapsed_seconds": time.perf_counter() - training_started,
        },
    )


if __name__ == "__main__":
    main()
