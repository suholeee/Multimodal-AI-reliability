# Data

This public release does not bundle any local datasets, checkpoints, or experiment logs.

The main experiments in this repository generate synthetic paired data at runtime from the polymer sandbox in `src/`. No download step is required to reproduce the released results.

Repository convention:

- keep generated or temporary local data out of git
- keep checkpoints out of git
- write derived artifacts only under `results/`

If you adapt this code to a real dataset, place raw data outside the repository or under this directory with a separate local-only ignore rule.
