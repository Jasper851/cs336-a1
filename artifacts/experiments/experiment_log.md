# Experiment Log

Record every completed, interrupted, and diverged run. Keep all comparison variables fixed except the
one named in the **Changed variable** column.

| Experiment | Goal | Changed variable | Training tokens | Device | Runtime | Final/best validation loss | Status | Run directory |
|---|---|---|---:|---|---:|---:|---|---|
| _example: lr_3e-4_ | _Tune learning rate_ | _max LR = 3e-4_ | _40,960,000_ | _MPS_ | _--_ | _--_ | _planned_ | _artifacts/experiments/lr_3e-4_ |

## Notes

For each experiment, briefly record:

- the hypothesis and conclusion;
- differences from the baseline configuration;
- divergence, interruption, bugs, or unusual behavior;
- the associated learning-curve figure and checkpoint, when applicable.
