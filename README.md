# Multimodal fusion for biological data: a study of failure modes

This repository releases a synthetic biology-focused multimodal learning study built around a shared latent chromatin-like system observed through paired image-like and genomic-like measurements. The motivating hypothesis was that combining modalities would improve classification accuracy and provide more robust decisions under disagreement. In the contradiction-focused setting studied here, that hypothesis did not hold cleanly: supervised fusion did not outperform the strongest unimodal baseline in the final observability pass, and zero-shot frontier agents improved classification only modestly while still failing to identify and act on modality contradiction. The result is a negative one, but a useful one: under these conditions, multimodal access did not by itself deliver reliable evidence reconciliation.

## Motivation

Combining biological modalities is attractive because different assays can expose different parts of the same underlying state. In practice, that creates a natural expectation that multimodal fusion should improve predictive accuracy, especially near ambiguous cases. This project tests that expectation in a controlled setting where both modalities are generated from the same latent polymer-based system, so disagreement can be studied without dataset-collection confounds.

## Approach

The repository simulates one latent chromatin-like polymer system and then observes it through two linked views:

- an image-like optical rendering
- a genomic contact-map representation

Small unimodal and fusion classifiers are trained on paired samples from these views for a binary condition-classification task over the synthetic system. The two public experiment entry points are:

- `scripts/finalize_reproducible_evaluation.py`: stable baseline evaluation in the weak-contradiction regime
- `scripts/run_v2_phase_diagram.py`: contradiction-focused V2 evaluation and validation pass
- `scripts/run_v3_agent_benchmark.py`: controlled zero-shot agent benchmark over public image/Hi-C evidence
- `scripts/summarize_v3_agent_runs.py`: aggregation of V3 agent runs across seeds, models, tools, and input conditions

The main implementation lives in `src/`:

- `polymer_generator.py`, `polymer_image_renderer.py`, `polymer_metrics.py`: synthetic data generation
- `ml_dataset.py`: paired dataset construction
- `ml_models.py`, `ml_train_eval.py`: model definitions and training loop
- `fusion_safety.py`, `fusion_debug_utils.py`, `contradiction_analysis.py`: fusion evaluation and disagreement analysis
- `v2_regime.py`, `v2_metrics.py`, `v2_policies.py`: stronger contradiction regime and routing/selective-policy analysis
- `v3_agent_dataset.py`, `v3_claude_client.py`, `v3_visual_tools.py`, `v3_scientific_tools.py`, `v3_agent_metrics.py`: controlled V3 agent benchmark

## Key Finding

### Supervised PyTorch Baselines

The stable baseline remains useful context, but it is not the headline result. In the five-seed baseline (`results/reports/final_results_summary.txt`), the best saved fusion variant (`fusion_frozen`) reached `0.760 +- 0.034` test accuracy versus `0.740 +- 0.047` for the genomic-only baseline and `0.655 +- 0.042` for the image-only baseline. That is a modest gain in a weak-contradiction regime, not strong evidence of robust multimodal complementarity.

The contradiction-focused V2 study is the main public result. In the saved contradiction sweep at `contradiction_strength=1.0`, image accuracy was `0.567`, genomic accuracy was `0.750`, and learned fusion accuracy was `0.767`, while the best full-coverage policy was `genomic_first` and policy gain over learned fusion was `0.000` (`results/v2/tables/v2_sweep_summary.csv`).

In the final observability pass, the hypothesis weakened further. At `contradiction_strength=1.0`, image accuracy was `0.583`, genomic accuracy was `0.550`, learned fusion accuracy was `0.517`, best-policy accuracy was also `0.517`, and the confident contradiction rate remained `0.000` at both `conf>=0.60` and `conf>=0.70` (`results/v2/reports/v2_validation_report.txt`).

Inference from those saved metrics: the learned fusion model did not extract a stable complementary signal under the tested contradiction regime and behaved similarly to a single-branch policy, especially `genomic_first`, rather than resolving edge-case disagreement in a balanced way.

### Zero-Shot V3 Agent Benchmark

V3 asks a different question: can frontier models classify and reconcile the same synthetic evidence zero-shot, without labeled examples or trained PyTorch weights? The final controlled V3 condition uses evidence-separation prompting, fixed non-leaky scientific tools, and matched seeds `21..40`.

Clean unimodal runs corrected an important contamination issue: modality-specific labels reported inside a both-modality response are not clean unimodal accuracies. The final clean unimodal results are:

| Model | Image only | Hi-C only | Both modalities | Contradiction acc | Action acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude Haiku 4.5, none | `60/120 = 0.500` | `66/120 = 0.550` | `60/120 = 0.500` | `64/120 = 0.533` | `64/120 = 0.533` |
| Claude Sonnet 4.6, none | `58/120 = 0.483` | `66/120 = 0.550` | `78/120 = 0.650` | `61/120 = 0.508` | `62/120 = 0.517` |
| Claude Sonnet 4.6, high | `63/120 = 0.525` | `67/120 = 0.558` | `69/120 = 0.575` | `56/120 = 0.467` | `64/120 = 0.533` |
| Claude Opus 4.7, none | `61/120 = 0.508` | `68/114 = 0.596` | `71/120 = 0.592` | `52/120 = 0.433` | `59/120 = 0.492` |

The strongest V3 classification result is `claude-sonnet-4-6 / none / evidence_separation / scientific / both_modalities` at `78/120 = 0.650`. Its paired gain over clean image-only is `+0.167` with approximate 95% interval `[+0.041, +0.292]`; its paired gain over clean Hi-C-only is `+0.100` with interval `[-0.002, +0.202]`.

That is meaningfully above chance, but it is not solved reconciliation. In the same best condition, true strong contradictions were predicted as `none` in `23/33` cases, true weak contradictions were predicted as `none` in `15/21` cases, and the model never predicted `use_image` even though `30` samples had `use_image` as the evaluator action.

The V3 result should therefore be read as a diagnostic zero-shot finding: models can use some biological priors from Hi-C-like evidence and sometimes benefit from both modalities, but they still mostly fail the harder task of detecting contradiction and choosing the right evidence-use policy. The curated final V3 summary is in `results/v3/final/`.

## What This Means

This is not evidence that multimodal fusion never works in biological ML. It is evidence that, on this synthesized shared-latent dataset and within the tested supervised architectures and zero-shot agent protocols, multimodal access did not reliably deliver the expected gains once contradiction was made central to the task. The project therefore supports a narrower and more defensible claim: multimodal fusion should not be assumed to improve reliability just because two views are available.

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

### Run a V3 agent benchmark

The V3 runner is mock-safe by default:

```bash
python scripts/run_v3_agent_benchmark.py --profile smoke --mock
```

A single real API V3 smoke run for the final scientific-tool condition looks like:

```bash
python scripts/run_v3_agent_benchmark.py \
  --profile smoke \
  --model claude-sonnet-4-6 \
  --input-condition both_modalities \
  --prompt-policy evidence_separation \
  --tool-level scientific \
  --thinking none \
  --seed 21 \
  --real-api \
  --confirm-api-call
```

The final V3 release repeats this over seeds `21..40`, model/thinking settings, and clean unimodal conditions (`--input-condition image_only` and `--input-condition hic_only`).

### Summarize the final V3 agent benchmark

The final V3 release summary combines final both-modality roots and the clean unimodal add-on:

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/final_model_sweep/scientific_20260505T201217Z \
  results/v3/scientific_tools/sonnet_evidence_sep_20260505T175124Z \
  results/v3/final_reasoning/sonnet_scientific_high_20260505T203622Z \
  results/v3/final_unimodal_clean/scientific_20260506T021839Z
```

See `docs/v3_agent_benchmark.md` for the exact run commands and `results/v3/final/README.md` for the curated final result.

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
- final V3 zero-shot agent summary:
  - `results/v3/final/README.md`
  - `results/v3/final/final_metrics.csv`
  - `results/v3/final/paired_deltas.csv`
  - `results/v3/final/prediction_bias_summary.csv`

## Limitations

- The dataset is synthetic.
- The release centers one architecture family and a limited set of routing policies.
- The strongest contradiction claim is supported by a saved single-seed sweep plus a later failed validation pass, not by a large benchmark campaign.
- The final observability pass is explicitly a negative result and should not be treated as a production-ready contradiction regime.
- V3 is zero-shot and prompt/tool dependent. It does not provide labeled exemplars to the model, and `thinking=high` is confounded with Anthropic's required `temperature=1.0`.
- V3 Opus Hi-C-only is missing one seed because the API run stopped early; it is reported as `n=114` and paired comparisons use the common sample set.

## Future Directions

- Real biological datasets are extremely noisy, and different experimental modalities often indicate different directions. Thus, this benchmark should be tested on real biological datasets. Possible candidates can be found through the 4DN Data Portal (https://data.4dnucleome.org).
- Develop better algorithms for efficient reconciliation: the central modeling question is how to process contradicting information without collapsing to one modality or over-trusting superficial agreement.
- V4 should move beyond one-shot classification toward a task-folder agent benchmark: give models a small public workspace of paired evidence files, require explicit evidence provenance, allow bounded tool use, and score not only final accuracy but also contradiction recall, action choice, abstention, and whether the model can explain which modality should be trusted.

## License

MIT. See `LICENSE`.
