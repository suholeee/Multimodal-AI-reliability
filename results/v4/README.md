# V4 Final Terminal-Agent Results

This directory contains the completed V4 Claude Code terminal-agent evaluation
for run `v4_compact_20260509T032302Z`.

The final scored comparison is:

```text
90 samples x 3 models x 3 input conditions = 810 scored predictions
```

All final outputs parsed cleanly: no malformed rows, no timeouts, and no
nonzero return codes in the scored set.

## Main Result

| model | both acc | image-only acc | Hi-C-only acc | contradiction acc | action acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| haiku | 0.622 | 0.444 | 0.500 | 0.311 | 0.322 |
| sonnet | 0.567 | 0.478 | 0.500 | 0.333 | 0.378 |
| opus | 0.522 | 0.556 | 0.500 | 0.322 | 0.333 |

The detailed interpretation is in
`results/v4/final_analysis_report.md`.

## Key Artifacts

- `final_analysis_report.md`: human-readable final analysis
- `evaluator/v4_compact_20260509T032302Z/hidden_manifest.jsonl`: hidden labels
  and evaluator metadata
- `evaluator/v4_compact_20260509T032302Z/contamination_audit.json`: public task
  leakage audit
- `evaluations/v4_compact_20260509T032302Z_full_clean/summary.csv`: aggregate
  metrics
- `evaluations/v4_compact_20260509T032302Z_full_clean/paired_deltas.csv`: paired
  multimodal-vs-unimodal deltas
- `evaluations/v4_compact_20260509T032302Z_full_clean/confusion.json`: contradiction
  and action confusion tables
- `evaluations/v4_compact_20260509T032302Z_full_clean/scored_predictions.jsonl`:
  scored per-sample outputs

Raw per-call run directories such as `results/v4/runs/` are local provenance
outputs and are intentionally not part of the public release commit.

## Interpretation

Haiku shows the strongest V4 classification result and the only clearly positive
paired both-minus-image gain. Sonnet shows smaller positive but uncertain
classification gains. Opus does not improve from both modalities.

The safety result is negative across all models: contradiction detection and
recommended evidence-use actions remain weak, and true strong contradictions
are usually predicted as `none`.
