from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation curves from experiment JSONL logs.")
    parser.add_argument("logs", type=Path, nargs="+", help="One or more metrics.jsonl files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/figures"))
    return parser.parse_args()


def read_validation_metrics(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps, times, losses = [], [], []
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("event") == "validation":
                steps.append(record["iteration"])
                times.append(record["elapsed_seconds"])
                losses.append(record["loss"])
    if not losses:
        raise ValueError(f"No validation records found in {path}.")
    return steps, times, losses


def save_plot(
    series: list[tuple[str, list[float], list[float]]],
    xlabel: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, x_values, losses in series:
        axis.plot(x_values, losses, marker="o", markersize=3, label=label)
    axis.set(xlabel=xlabel, ylabel="Validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_step, by_time = [], []
    for log_path in args.logs:
        steps, times, losses = read_validation_metrics(log_path)
        label = log_path.parent.name
        by_step.append((label, steps, losses))
        by_time.append((label, times, losses))

    save_plot(by_step, "Gradient step", args.output_dir / "loss_vs_steps.png")
    save_plot(by_time, "Wall-clock time (seconds)", args.output_dir / "loss_vs_time.png")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
