# Multimodal fusion for biological data: a study of failure modes

This repository releases a synthetic biology-focused multimodal learning study built around a shared latent chromatin-like system observed through paired image-like and genomic-like measurements. The motivating hypothesis was that combining modalities would improve classification accuracy and provide more robust decisions under disagreement. In the contradiction-focused setting studied here, that hypothesis did not hold cleanly: fusion did not outperform the strongest unimodal baseline in the final observability pass, and the learned system behaved consistently with collapse toward the stronger modality rather than extracting stable complementary signal. The result is a negative one, but a useful one: under these conditions, naive multimodal fusion did not deliver the expected reliability gain.

## Motivation

Combining biological modalities is attractive because different assays can expose different parts of the same underlying state. In practice, that creates a natural expectation that multimodal fusion should improve predictive accuracy, especially near ambiguous cases. This project tests that expectation in a controlled setting where both modalities are generated from the same latent polymer-based system, so disagreement can be studied without dataset-collection confounds.

## Approach

The repository simulates one latent chromatin-like polymer system and then observes it through two linked views:

- an image-like optical rendering
- a genomic contact-map representation

Small unimodal and fusion classifiers are trained on paired samples from these views for a binary condition-classification task over the synthetic system. The two public experiment entry points are:

- `scripts/finalize_reproducible_evaluation.py`: stable baseline evaluation in the weak-contradiction regime
- `scripts/run_v2_phase_diagram.py`: contradiction-focused V2 evaluation and validation pass

The main implementation lives in `src/`:

- `polymer_generator.py`, `polymer_image_renderer.py`, `polymer_metrics.py`: synthetic data generation
- `ml_dataset.py`: paired dataset construction
- `ml_models.py`, `ml_train_eval.py`: model definitions and training loop
- `fusion_safety.py`, `fusion_debug_utils.py`, `contradiction_analysis.py`: fusion evaluation and disagreement analysis
- `v2_regime.py`, `v2_metrics.py`, `v2_policies.py`: stronger contradiction regime and routing/selective-policy analysis

## Key Finding

The stable baseline remains useful context, but it is not the headline result. In the five-seed baseline (`results/reports/final_results_summary.txt`), the best saved fusion variant (`fusion_frozen`) reached `0.760 +- 0.034` test accuracy versus `0.740 +- 0.047` for the genomic-only baseline and `0.655 +- 0.042` for the image-only baseline. That is a modest gain in a weak-contradiction regime, not strong evidence of robust multimodal complementarity.

The contradiction-focused V2 study is the main public result. In the saved contradiction sweep at `contradiction_strength=1.0`, image accuracy was `0.567`, genomic accuracy was `0.750`, and learned fusion accuracy was `0.767`, while the best full-coverage policy was `genomic_first` and policy gain over learned fusion was `0.000` (`results/v2/tables/v2_sweep_summary.csv`).

In the final observability pass, the hypothesis weakened further. At `contradiction_strength=1.0`, image accuracy was `0.583`, genomic accuracy was `0.550`, learned fusion accuracy was `0.517`, best-policy accuracy was also `0.517`, and the confident contradiction rate remained `0.000` at both `conf>=0.60` and `conf>=0.70` (`results/v2/reports/v2_validation_report.txt`).

Inference from those saved metrics: the learned fusion model did not extract a stable complementary signal under the tested contradiction regime and behaved similarly to a single-branch policy, especially `genomic_first`, rather than resolving edge-case disagreement in a balanced way.

## What This Means

This is not evidence that multimodal fusion never works in biological ML. It is evidence that, on this synthesized shared-latent dataset and within this architecture family, naive fusion did not reliably deliver the expected gains once contradiction was made central to the task. The project therefore supports a narrower and more defensible claim: multimodal fusion should not be assumed to improve reliability just because two views are available.

## What's Needed Next

- evaluation on real biological datasets with measured contradiction rates between modalities
- ablations that isolate which architectural components mediate branch dominance or modality collapse
- fusion strategies that explicitly model contradiction rather than assuming complementarity
- direct measurement of when fused predictions should be deferred, abstained, or flagged

## How to Reproduce

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

No external dataset is bundled. The released experiments generate synthetic paired data at runtime from the polymer sandbox. See `data/README.md` for the expected repository convention.

### Run the stable baseline

```bash
python scripts/finalize_reproducible_evaluation.py
```

This writes baseline artifacts under `results/final/` and `results/reports/`.

### Run the contradiction-focused V2 evaluation

```bash
python scripts/run_v2_phase_diagram.py --profile default --no-show
```

For a faster smoke run:

```bash
python scripts/run_v2_phase_diagram.py --profile debug --no-show
```

This writes figures, reports, and tables under `results/v2/`.

## Included Results

- baseline summary: `results/reports/final_results_summary.txt`
- baseline plots:
  - `results/final/final_accuracy_summary.png`
  - `results/final/final_contradiction_summary.png`
- contradiction-focused figure: `results/v2/figures/v2_contradiction_sweep.png`
- contradiction-focused reports:
  - `results/v2/reports/v2_main_findings.txt`
  - `results/v2/reports/v2_validation_report.txt`
  - `results/v2/reports/v2_observability_pass_negative_result.txt`
- contradiction-focused tables:
  - `results/v2/tables/v2_sweep_metrics.csv`
  - `results/v2/tables/v2_sweep_summary.csv`
  - `results/v2/tables/v2_validation_metrics.csv`

## Limitations

- The dataset is synthetic.
- The release centers one architecture family and a limited set of routing policies.
- The strongest contradiction claim is supported by a saved single-seed sweep plus a later failed validation pass, not by a large benchmark campaign.
- The final observability pass is explicitly a negative result and should not be treated as a production-ready contradiction regime.

## License

MIT. See `LICENSE`.
