# Claude Code Project Context

This project is a synthetic multimodal reliability study for biological-style data. The key research question is not "can fusion raise accuracy?" but "can a model reconcile image-like and genomic-like evidence when the modalities agree, disagree, or differ in reliability?"

## What Matters

- Keep the negative-result framing intact.
- Treat contradiction observability as a first-class metric.
- Avoid optimizing only for test accuracy.
- Prefer small falsifiable changes over broad rewrites.
- Preserve reproducibility and validation gates.

## Main Files

- `src/v2_regime.py`: contradiction-aware synthetic regime.
- `src/ml_models.py`: unimodal and fusion classifiers.
- `src/fusion_debug_utils.py`: learned and late fusion variants plus branch diagnostics.
- `src/fusion_safety.py`: output-only disagreement and abstention signals.
- `src/v2_metrics.py`: calibration, regret, disagreement, and strategy summaries.
- `src/v2_policies.py`: routing and abstention policies.
- `scripts/run_v2_phase_diagram.py`: V2 validation gate and sweep.
- `scripts/finalize_reproducible_evaluation.py`: five-seed baseline.

## Good Claude Code Tasks

- Add a test for one metric or policy.
- Add a new contradiction regime with explicit ground-truth reliability labels.
- Add calibration or uncertainty estimation around existing model outputs.
- Add a reconciliation policy that can choose image, genomic, fusion, or abstain.
- Extend reports with per-regime oracle regret, branch dominance, and risk-coverage curves.

## Commands

Smoke run:

```bash
python scripts/run_v2_phase_diagram.py --profile debug --no-show
```

Baseline run:

```bash
python scripts/finalize_reproducible_evaluation.py
```

Default V2 run:

```bash
python scripts/run_v2_phase_diagram.py --profile default --no-show
```

## Definition of Done

A change is not complete until the agent reports:

- what scientific hypothesis changed
- which files changed
- which command was run
- image/genomic/late/learned/policy accuracy
- confident contradiction rate at `conf>=0.70`
- oracle regret or oracle gap
- selective accuracy or abstention behavior
- whether the validation gate passed or failed
