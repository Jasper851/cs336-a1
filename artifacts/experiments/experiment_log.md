# Experiment Log

Record every completed, interrupted, and diverged run. Keep all comparison variables fixed except the
one named in the **Changed variable** column.

| Experiment | Goal | Changed variable | Training tokens | Device | Runtime | Final/best validation loss | Status | Run directory |
|---|---|---|---:|---|---:|---:|---|---|
| `ts_mps_calibration` | Verify MPS training and batch 32 | calibration run | 819,200 | MPS | 2m 26s | 4.2367 | complete; no OOM/NaN | `artifacts/experiments/ts_mps_calibration` |
| `ts_mps_lr_1e-4` | LR coarse sweep | max LR = `1e-4` | 4,096,000 | MPS | 12m 49s | 3.5546 | complete; too slow | `artifacts/experiments/ts_mps_lr_1e-4` |
| `ts_mps_lr_3e-4` | LR coarse sweep | max LR = `3e-4` | 4,096,000 | MPS | 13m 46s | 2.9028 | complete | `artifacts/experiments/ts_mps_lr_3e-4` |
| `ts_mps_lr_1e-3` | LR coarse sweep | max LR = `1e-3` | 4,096,000 | MPS | 13m 27s | 2.4488 | coarse-sweep best | `artifacts/experiments/ts_mps_lr_1e-3` |
| `ts_mps_lr_3e-3` | LR coarse sweep | max LR = `3e-3` | 4,096,000 | MPS | 14m 03s | 2.4763 | complete; close second | `artifacts/experiments/ts_mps_lr_3e-3` |
| `ts_mps_lr_1e-2` | LR coarse sweep | max LR = `1e-2` | 4,096,000 | MPS | 13m 50s | 3.1617 | complete; degraded | `artifacts/experiments/ts_mps_lr_1e-2` |
| `ts_mps_lr_3e-2` | LR coarse sweep | max LR = `3e-2` | 4,096,000 | MPS | 13m 54s | 3.8078 | complete; degraded | `artifacts/experiments/ts_mps_lr_3e-2` |
| `ts_mps_lr_1e-1` | LR coarse sweep | max LR = `1e-1` | 4,096,000 | MPS | 14m 00s | 4.1792 | complete; severely degraded | `artifacts/experiments/ts_mps_lr_1e-1` |
| `ts_mps_lr_3e-1` | LR coarse sweep | max LR = `3e-1` | 4,096,000 | MPS | 14m 32s | 4.3285 | complete; unstable | `artifacts/experiments/ts_mps_lr_3e-1` |
| `ts_mps_lr_1` | LR coarse sweep | max LR = `1` | 4,096,000 | MPS | 13m 43s | 7.0105 | diverged; partial late recovery | `artifacts/experiments/ts_mps_lr_1` |
| `ts_mps_lr_fine_1e-3` | Select best LR at 1500 steps | max LR = `1e-3` | 12,288,000 | MPS | 39m 02s | **1.9502** | selected best LR | `artifacts/experiments/ts_mps_lr_fine_1e-3` |
| `ts_mps_lr_fine_2e-3` | Select best LR at 1500 steps | max LR = `2e-3` | 12,288,000 | MPS | 42m 29s | 1.9675 | complete; close second | `artifacts/experiments/ts_mps_lr_fine_2e-3` |
| `ts_mps_lr_fine_3e-3` | Select best LR at 1500 steps | max LR = `3e-3` | 12,288,000 | MPS | 41m 50s | 2.1550 | complete | `artifacts/experiments/ts_mps_lr_fine_3e-3` |
| `ts_mps_bs_memory_probe_32` | Probe MPS batch limit | batch size = 32 | 163,840 | MPS | 0m 43s | 4.9658 | complete | `artifacts/experiments/ts_mps_bs_memory_probe_32` |
| `ts_mps_bs_memory_probe_64` | Probe MPS batch limit | batch size = 64 | 327,680 | MPS | 1m 44s | 4.9194 | complete; maximum tested success | `artifacts/experiments/ts_mps_bs_memory_probe_64` |
| `ts_mps_bs_memory_probe_128` | Probe MPS batch limit | batch size = 128 | 0 | MPS | -- | -- | OOM before first step | `artifacts/experiments/ts_mps_bs_memory_probe_128` |
| `ts_mps_bs_1` | Compare batch sizes at fixed tokens | batch size = 1 | 4,194,304 | MPS | 19m 37s | 2.8564 / 2.3465 | complete; noisy validation | `artifacts/experiments/ts_mps_bs_1` |
| `ts_mps_bs_8` | Compare batch sizes at fixed tokens | batch size = 8 | 4,194,304 | MPS | 14m 18s | **2.3101 / 2.2823** | complete; best final loss | `artifacts/experiments/ts_mps_bs_8` |
| `ts_mps_bs_32` | Compare batch sizes at fixed tokens | batch size = 32 | 4,194,304 | MPS | 14m 46s | 2.4659 / 2.4565 | complete; highest throughput | `artifacts/experiments/ts_mps_bs_32` |
| `ts_mps_bs_64` | Compare batch sizes at fixed tokens | batch size = 64 | 4,194,304 | MPS | 18m 33s | 2.6883 / 2.6883 | complete; no speed/loss benefit | `artifacts/experiments/ts_mps_bs_64` |
| `ts_mps_best_full` | Final low-resource training | batch 32, max LR = `1e-3` | 40,960,000 | MPS | ~2h 19m | 1.6264 / **1.5949** | complete; resumed at step 2524 | `artifacts/experiments/ts_mps_best_full` |

## Notes

For each experiment, briefly record:

- the hypothesis and conclusion;
- differences from the baseline configuration;
- divergence, interruption, bugs, or unusual behavior;
- the associated learning-curve figure and checkpoint, when applicable.

### Learning-rate coarse sweep

All coarse-sweep runs used the same model, seed, batch size 32, context length 256, and 500-step budget.
The best final validation loss was 2.4488 at max LR `1e-3`; `3e-3` was close at 2.4763. Performance
degraded from `1e-2` upward. Max LR `1` clearly diverged during its high-LR phase: validation loss reached
1022.91 at step 100, then partially recovered only as the cosine schedule reduced the LR. The observed
divergence boundary is therefore between `3e-1` and `1`.

Curve: `artifacts/experiments/figures/tinystories_lr_probe/loss_vs_steps.png`.

### Best learning-rate selection

The three fine runs used the same configuration and 1500-step/12.288M-token budget. Max LR `2e-3` was
slightly better at step 100, but `1e-3` was consistently better from step 300 through step 1500. Final
validation losses were 1.9502 (`1e-3`), 1.9675 (`2e-3`), and 2.1550 (`3e-3`). Select max LR `1e-3`
and min LR `1e-4` for subsequent experiments and the full training run.

Curve: `artifacts/experiments/figures/tinystories_lr_fine/loss_vs_steps.png`.

### Batch-size experiment

Batch sizes 1, 8, 32, and 64 each processed exactly 4,194,304 training tokens with max LR `1e-3`.
Batch 128 failed before its first optimizer step with an MPS out-of-memory error, so 64 is the largest
successful standard batch size. Batch 8 had the best observed final validation loss (2.3101) and shortest
runtime (14m 18s); batch 32 had the highest average throughput (about 5,114 tokens/s). Batch 64 was both
slower and worse at the fixed token budget, so it should receive a targeted LR retune before the comparison
is finalized.

Validation used 10 batches at the current training batch size. Consequently, each estimate covered 2,560
tokens for batch 1 but 163,840 tokens for batch 64. This explains much of the visible noise in the batch-1
curve, so its individual validation points and final/best gap should not be over-interpreted. The existing
step-axis plot also compares different amounts of processed data at the same x-coordinate; final endpoints
are comparable because all runs used the same total token budget, but a processed-token-axis plot is better
for the final report.

Curves: `artifacts/experiments/figures/tinystories_batch_size/loss_vs_steps.png` and
`artifacts/experiments/figures/tinystories_batch_size/loss_vs_time.png`.

The batch-size experiment was closed without an additional LR sweep. Batch 8 gave the best short-budget
final loss, but batch 32 was retained for final training because it had the highest throughput and was the
configuration used to tune the selected LR. Batch 64 did not provide a speed or loss advantage.

### Final low-resource training

The final model used batch size 32, max LR `1e-3`, min LR `1e-4`, and 5000 steps for exactly 40,960,000
training tokens. The first session was interrupted after 2524 completed updates; the saved model, optimizer,
and iteration state were restored and training completed normally. Final validation loss was 1.6264, with
the best observed validation loss of 1.5949 at step 4800. This exceeds the low-resource target of validation
loss at most 2.00. The generation checkpoint is
`artifacts/experiments/ts_mps_best_full/checkpoint.pt`.

### Generation experiment

All samples used the prompt `Once upon a time`, seed 42, and the final `ts_mps_best_full` checkpoint.
Each run requested at most 256 new tokens and stopped when it generated `<|endoftext|>`.

| Sample | Temperature | Top-p | Approx. new tokens | Assessment | Output |
|---|---:|---:|---:|---|---|
| Balanced | 0.8 | 0.90 | 173 | Fluent overall, but contains pronoun, ownership, and dialogue inconsistencies | `artifacts/experiments/ts_mps_best_full/generate_balanced.txt` |
| Conservative | 0.6 | 0.90 | 137 | Most coherent and complete; less varied and somewhat formulaic | `artifacts/experiments/ts_mps_best_full/generate_conservative.txt` |
| Creative | 1.0 | 0.95 | 215 | Longest and more varied, but more repetitive and logically inconsistent | `artifacts/experiments/ts_mps_best_full/generate_creative.txt` |

The conservative sample is the strongest fluency example. Generation quality is helped by the final
validation loss of 1.6264 and the 40.96M-token training budget. Temperature and top-p control the other
main tradeoff: lowering temperature improves consistency but reduces variety, while higher temperature
and top-p increase diversity along with contradictions and repetition.
> **Pending:** Due to current computational resource constraints, Experiments 7.3 (ablations), 7.4 (training on OpenWebText), and 7.5 (custom modifications and leaderboard submission) cannot be completed at this time and will be added once suitable resources become available.

