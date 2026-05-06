# Agent Instructions

This repository studies multimodal AI reliability under image-like and genomic-like experimental evidence. Treat the current public result as a negative/diagnostic finding, not as a solved fusion benchmark.

## Scientific Goal

Improve AI systems that reconcile experimental modalities when:

- modalities agree but carry different reliability or uncertainty
- modalities disagree because each assay sees a different part of the latent biology
- one modality is misleading because of noise, artifacts, missingness, or distribution shift
- the correct action is to abstain, request another assay, or report unresolved evidence

Prefer changes that make this distinction explicit in data generation, model outputs, and evaluation.

## Current Repo Shape

- Synthetic paired data is generated in `src/ml_dataset.py` and `src/v2_regime.py`.
- Polymer simulation and rendering live in `src/polymer_generator.py`, `src/polymer_image_renderer.py`, and `src/polymer_metrics.py`.
- Models and training live in `src/ml_models.py`, `src/ml_train_eval.py`, and `src/fusion_debug_utils.py`.
- Contradiction and policy evaluation live in `src/contradiction_analysis.py`, `src/fusion_safety.py`, `src/v2_metrics.py`, and `src/v2_policies.py`.
- Reproducible entry points are `scripts/finalize_reproducible_evaluation.py` and `scripts/run_v2_phase_diagram.py`.
- Saved negative-result reports are under `results/reports/` and `results/v2/reports/`.

## Development Rules

- Preserve negative results. Do not tune until the contradiction problem disappears.
- Keep validation/test separation clean. Tune routing, calibration, and thresholds on validation data only.
- Report every new method against unimodal, late-fusion, learned-fusion, routing, oracle-branch, and abstention baselines.
- Stratify metrics by agreement, disagreement, structural contradiction, and confidence bands.
- If adding synthetic regimes, expose latent ground truth needed to evaluate reconciliation, not just final classification.
- If adding real datasets, keep raw data out of git and document download/provenance steps.
- Add tests for new metrics, policy logic, and data-generation invariants.

## Required Checks

For quick verification:

```bash
python scripts/run_v2_phase_diagram.py --profile debug --no-show
```

For stronger validation before treating a change as a candidate result:

```bash
python scripts/finalize_reproducible_evaluation.py
python scripts/run_v2_phase_diagram.py --profile default --no-show
```

If the V2 validation gate fails, preserve the failure report and explain what failed. Do not silently proceed as if the full sweep is meaningful.

## Agent Workflow

When using Codex, Claude Code, or another coding agent, split work into these bounded tasks:

1. Diagnose the current failure mode from saved reports and per-sample tables.
2. Propose one falsifiable reconciliation hypothesis.
3. Modify one bounded layer: regime, model, policy, metric, or report.
4. Run the debug profile.
5. Inspect whether contradiction observability, calibration, oracle regret, and selective risk improved.
6. Write a short result note under `results/` or `docs/` before attempting a larger sweep.

Useful agent prompt:

```text
Investigate this repo as a multimodal evidence-reconciliation benchmark. Preserve the negative-result framing. Add one bounded change that improves measurement or modeling of modality disagreement, then run the debug V2 sweep and summarize accuracy, confident contradiction rate, oracle regret, and selective accuracy.
```
