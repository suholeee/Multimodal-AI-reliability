# Final V3 Zero-Shot Agent Results

This is the release-facing summary for the V3 controlled API benchmark. V3
tests zero-shot frontier model interpretation of synthetic polymer image and
Hi-C evidence. No labeled normal/cancer exemplars are shown to the model.

## Final Raw Inputs

The final summary combines these raw roots:

- `results/v3/final_model_sweep/scientific_20260505T201217Z`
- `results/v3/scientific_tools/sonnet_evidence_sep_20260505T175124Z`
- `results/v3/final_reasoning/sonnet_scientific_high_20260505T203622Z`
- `results/v3/final_unimodal_clean/scientific_20260506T021839Z`

All release conditions use:

- `prompt_policy=evidence_separation`
- `tool_level=scientific`
- `profile=smoke`
- seeds `21..40`

Opus Hi-C-only has `n=114` rather than `n=120` because seed `40` did not finish
before the API run stopped. It is retained with explicit sample count.

## Headline Metrics

| Model | Image only | Hi-C only | Both modalities | Contradiction acc | Action acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude Haiku 4.5, none | `60/120 = 0.500` | `66/120 = 0.550` | `60/120 = 0.500` | `64/120 = 0.533` | `64/120 = 0.533` |
| Claude Opus 4.7, none | `61/120 = 0.508` | `68/114 = 0.596` | `71/120 = 0.592` | `52/120 = 0.433` | `59/120 = 0.492` |
| Claude Sonnet 4.6, high | `63/120 = 0.525` | `67/120 = 0.558` | `69/120 = 0.575` | `56/120 = 0.467` | `64/120 = 0.533` |
| Claude Sonnet 4.6, none | `58/120 = 0.483` | `66/120 = 0.550` | `78/120 = 0.650` | `61/120 = 0.508` | `62/120 = 0.517` |

The strongest classification condition is:

```text
claude-sonnet-4-6 / none / evidence_separation / scientific / both_modalities
accuracy = 78/120 = 0.650
contradiction accuracy = 61/120 = 0.508
action accuracy = 62/120 = 0.517
```

Paired classification deltas for Sonnet none:

- both-modalities minus clean image-only: `+0.167`, approximate 95% interval `[+0.041, +0.292]`, `n=120`
- both-modalities minus clean Hi-C-only: `+0.100`, approximate 95% interval `[-0.002, +0.202]`, `n=120`

## Interpretation

The clean unimodal add-on changes the result presentation:

- Image-only zero-shot accuracy is near chance for all models.
- Hi-C-only zero-shot accuracy has modest signal, strongest for Opus at `68/114 = 0.596`.
- Sonnet none is the only condition with a clear both-modality classification gain over clean image-only.
- Opus both-modality accuracy is essentially its clean Hi-C-only accuracy on common samples.
- Haiku mostly collapses toward `normal` and `use_both`.

The classification signal is meaningful but the reconciliation result is still
negative. In the best classification condition, true strong contradictions are
predicted as `none` in `23/33` cases and true weak contradictions in `15/21`
cases. Sonnet none also never predicts `use_image`, despite `30` true
`use_image` cases.

## Release Tables

- `final_metrics.csv`: pooled and per-run final metrics.
- `paired_deltas.csv`: matched `(seed, sample_id)` classification deltas.
- `prediction_bias_summary.csv`: label-bias summaries for clean unimodal and both-modality runs.

## Reproduce Summary

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/final_model_sweep/scientific_20260505T201217Z \
  results/v3/scientific_tools/sonnet_evidence_sep_20260505T175124Z \
  results/v3/final_reasoning/sonnet_scientific_high_20260505T203622Z \
  results/v3/final_unimodal_clean/scientific_20260506T021839Z
```
