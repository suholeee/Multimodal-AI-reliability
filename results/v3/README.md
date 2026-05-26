# V3 Result Directory Map

This directory contains controlled zero-shot agent benchmark outputs. The raw
API runs are intentionally kept, but the release-facing summary is separated
from exploratory and test runs.

## Release Summary

- `final/`: curated V3 final result tables and interpretation for release.

Start with `final/README.md`.

## Final Raw Roots

- `final_model_sweep/scientific_20260505T201217Z`: final Haiku and Opus scientific both-modality model sweep.
- `scientific_tools/sonnet_evidence_sep_20260505T175124Z`: final Sonnet none scientific both-modality sweep.
- `final_reasoning/sonnet_scientific_high_20260505T203622Z`: final Sonnet high scientific both-modality sweep.
- `final_unimodal_clean/scientific_20260506T021839Z`: clean image-only and Hi-C-only add-on for Haiku, Opus, Sonnet none, and Sonnet high.

## Exploratory / Gate Runs

- `runs/`: early smoke and API validation runs.
- `stability/`: repeated-run stability checks.
- `performance/`: earlier Sonnet visual-tool performance runs.
- `prompt_ablation/`: Sonnet evidence-separation prompt ablations.
- `python_tools/`: Level 3 sandboxed Python gates and partial experiments.

These runs are useful for provenance, but they are not the final V3 headline
unless explicitly referenced from `final/`.
