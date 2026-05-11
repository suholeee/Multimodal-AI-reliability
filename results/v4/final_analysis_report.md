# V4 Terminal-Agent Final Analysis Report

Run ID: `v4_compact_20260509T032302Z`

This report summarizes the completed V4 Claude Code terminal-agent benchmark.
V4 asks whether a terminal agent, operating over clean public task folders, can
classify synthetic polymer-derived normal/cancer samples and reconcile
contradictory evidence between a polymer image and a Hi-C contact map.

## Evaluation Design

The compact final set used a frozen hidden manifest with `90` public sample IDs:

| true contradiction status | normal | cancer | total |
| --- | ---: | ---: | ---: |
| none | 15 | 15 | 30 |
| weak | 15 | 15 | 30 |
| strong | 15 | 15 | 30 |
| total | 45 | 45 | 90 |

Three Claude Code model aliases were run at `--effort high`:

- `haiku`
- `sonnet`
- `opus`

Each model was evaluated under three input conditions:

- `both_modalities`: polymer image plus Hi-C map, with reconciliation required
- `image_only`: polymer image only
- `hic_only`: Hi-C map only

The completed evaluation therefore contains:

```text
90 samples x 3 models x 3 conditions = 810 scored predictions
```

The public task folders contained only visible evidence, instructions, schema,
and a task-local work directory. Hidden labels, contradiction status,
recommended action, source indices, latent variables, and reliability proxies
were kept in `results/v4/evaluator/v4_compact_20260509T032302Z/hidden_manifest.jsonl`
and used only by the scorer.

## Artifact Map

Committed primary outputs:

- `results/v4/evaluations/v4_compact_20260509T032302Z_full_clean/summary.csv`
- `results/v4/evaluations/v4_compact_20260509T032302Z_full_clean/paired_deltas.csv`
- `results/v4/evaluations/v4_compact_20260509T032302Z_full_clean/confusion.json`
- `results/v4/evaluations/v4_compact_20260509T032302Z_full_clean/scored_predictions.jsonl`
- `results/v4/evaluations/v4_compact_20260509T032302Z_full_clean/report.txt`
- `results/v4/evaluator/v4_compact_20260509T032302Z/hidden_manifest.jsonl`
- `results/v4/evaluator/v4_compact_20260509T032302Z/selected_public_assets/`

Local provenance outputs:

- `results/v4/runs/v4_compact_20260509T032302Z_final_both/`
- `results/v4/runs/v4_compact_20260509T032302Z_unimodal/`

The raw run directories contain per-call stdout, stderr, and metadata. They are
useful for local audit, but they follow the repository's existing `runs/` ignore
rule and are not required to reproduce the scored summaries committed above.

## Completion And Data Quality

The final run is complete and parse-clean.

| condition group | expected | complete |
| --- | ---: | ---: |
| both modalities | 270 | 270 |
| image only plus Hi-C only | 540 | 540 |
| total scored rows | 810 | 810 |

Quality checks:

- malformed output fraction: `0.000` for every model and condition
- timeout rate: `0.000` for every model and condition
- nonzero return-code rate: `0.000` for every model and condition
- total reported cost: `$96.46`

This means the final comparison is not limited by parsing failures or incomplete
conditions.

## Main Results

| model | both acc | image-only acc | Hi-C-only acc | contradiction acc | action acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| haiku | 0.622 | 0.444 | 0.500 | 0.311 | 0.322 |
| sonnet | 0.567 | 0.478 | 0.500 | 0.333 | 0.378 |
| opus | 0.522 | 0.556 | 0.500 | 0.322 | 0.333 |

Interpretation:

- `haiku` has the strongest both-modality classification result: `56/90 = 0.622`.
- `sonnet` is above chance but below the rough `55/90 = 0.611` threshold noted in the protocol for a single model's classification result against chance.
- `opus` does not improve under both modalities; its image-only accuracy is higher than its both-modality accuracy.
- all three models remain weak at contradiction detection and evidence-use action selection.

## Paired Multimodal Deltas

The clean unimodal add-on allows paired comparisons on the same `90` sample IDs.

| model | paired comparison | delta | approx 95% interval |
| --- | --- | ---: | ---: |
| haiku | both - image | +0.178 | `[+0.044, +0.311]` |
| haiku | both - Hi-C | +0.122 | `[-0.022, +0.267]` |
| sonnet | both - image | +0.089 | `[-0.034, +0.211]` |
| sonnet | both - Hi-C | +0.067 | `[-0.030, +0.164]` |
| opus | both - image | -0.033 | `[-0.159, +0.092]` |
| opus | both - Hi-C | +0.022 | `[-0.047, +0.091]` |

Only the `haiku` both-minus-image delta is clearly positive under this
approximate interval. The `haiku` both-minus-Hi-C delta is numerically positive
but crosses zero. The `sonnet` gains are also positive but uncertain. `opus`
does not support a multimodal gain claim.

The narrowest defensible V4 classification claim is therefore:

> In this terminal-agent setup, Haiku showed a clean paired classification gain
> from seeing both modalities relative to image-only evidence, but the broader
> multimodal advantage was not consistent across models or across both unimodal
> baselines.

## Why Hi-C-Only Accuracy Is Exactly 50%

Each Hi-C-only condition has exactly `45` cancer and `45` normal labels. All
three models landed at exactly `45/90` correct, but by different prediction
biases:

| model | predicted cancer | predicted normal | correct |
| --- | ---: | ---: | ---: |
| haiku | 26 | 64 | 45 |
| sonnet | 82 | 8 | 45 |
| opus | 84 | 6 | 45 |

The exact `0.500` result is not a rounding artifact. The predictions were
effectively uncorrelated with the hidden labels on a balanced label set.
`sonnet` and `opus` mostly called Hi-C-only samples `cancer`, while `haiku`
favored `normal`; neither bias tracked the true labels.

This is a central negative result: the terminal agents did not extract reliable
zero-shot label signal from the clean Hi-C-only task folders in this V4 setup.

## Prediction Bias And Confidence

Both-modality prediction counts:

| model | predicted normal | predicted cancer | mean confidence | high-conf wrong, conf >= 0.8 |
| --- | ---: | ---: | ---: | ---: |
| haiku | 49 | 41 | 0.746 | 12 |
| sonnet | 18 | 72 | 0.669 | 1 |
| opus | 14 | 76 | 0.650 | 1 |

The models show different failure modes:

- `haiku` is the most accurate classifier in the both-modality condition, but it
  is also the most overconfident when wrong, with `12/90` high-confidence wrong
  predictions.
- `sonnet` and `opus` strongly bias toward `cancer` in both-modality and
  Hi-C-only settings.
- `opus` has the best image-only result (`0.556`) but loses that advantage when
  both modalities are present.

Calibration metrics from `summary.csv` are consistent with this:

| model | both ECE | image-only ECE | Hi-C-only ECE |
| --- | ---: | ---: | ---: |
| haiku | 0.127 | 0.285 | 0.271 |
| sonnet | 0.102 | 0.133 | 0.237 |
| opus | 0.182 | 0.012 | 0.201 |

`opus` image-only is the best-calibrated single condition, but this does not
translate into better both-modality reconciliation.

## Contradiction Detection

The contradiction task is the main safety-relevant part of V4. It is also where
the run is most clearly negative.

Both-modality contradiction accuracy by true contradiction status:

| model | true none | true weak | true strong |
| --- | ---: | ---: | ---: |
| haiku | 0.767 | 0.167 | 0.000 |
| sonnet | 0.833 | 0.167 | 0.000 |
| opus | 0.800 | 0.133 | 0.033 |

Strong contradictions were usually missed:

| model | true strong predicted none | true strong predicted weak | true strong predicted strong |
| --- | ---: | ---: | ---: |
| haiku | 27 | 3 | 0 |
| sonnet | 25 | 5 | 0 |
| opus | 25 | 4 | 1 |

This pattern is important. The agents are not merely confusing weak and strong
contradiction. They most often collapse true contradiction into `none`, which is
the failure mode most relevant to unsafe evidence reconciliation.

## Recommended Action Accuracy

Recommended action accuracy is also near chance:

| model | action accuracy |
| --- | ---: |
| haiku | 0.322 |
| sonnet | 0.378 |
| opus | 0.333 |

The dominant action prediction is `use_both`, even for samples where the hidden
evaluator action is `use_image` or `use_hic`.

Both-modality predicted actions:

| model | use_both | use_hic | use_image | abstain |
| --- | ---: | ---: | ---: | ---: |
| haiku | 70 | 14 | 2 | 4 |
| sonnet | 73 | 15 | 2 | 0 |
| opus | 73 | 17 | 0 | 0 |

The hidden task set is balanced by label and contradiction status, while the
recommended actions are induced by the evaluator policy and reliability proxies.
Across that action mix, the agents mostly avoid `use_image` and rarely abstain.
This means they do not reliably use contradiction evidence to decide when one
modality should be trusted over the other.

## Cost And Latency

| model | both cost | image-only cost | Hi-C-only cost | total cost |
| --- | ---: | ---: | ---: | ---: |
| haiku | `$2.81` | `$2.55` | `$4.90` | `$10.26` |
| sonnet | `$16.40` | `$11.89` | `$17.76` | `$46.05` |
| opus | `$13.77` | `$12.92` | `$13.46` | `$40.14` |

Mean latency by condition:

| model | both | image-only | Hi-C-only |
| --- | ---: | ---: | ---: |
| haiku | 30.2s | 25.2s | 50.3s |
| sonnet | 124.0s | 91.5s | 134.5s |
| opus | 37.9s | 32.7s | 33.6s |

`haiku` is much cheaper and faster than `sonnet` while also producing the best
both-modality classification result in this run. However, that efficiency does
not solve contradiction or action selection.

## Relationship To V3

V3 and V4 are not identical evaluations:

- V3 used a controlled API harness with fixed tools and a stricter evaluator-owned
  tool loop.
- V4 used Claude Code as a terminal agent over public task folders, allowing
  more autonomous file reading and shell/Python strategy.

The broad conclusion is stable across both:

- both-modality access can produce modest classification gains in some settings
- clean unimodal controls are necessary before claiming multimodal improvement
- contradiction detection and recommended evidence-use actions remain weak
- models tend to hallucinate consistency or default to using both modalities

V4 sharpened the result by showing that a more agentic terminal workflow did not
fix the reconciliation problem.

## Main Conclusion

The completed V4 run supports a mixed but mostly negative reliability conclusion.

Classification:

- `haiku` benefits from both modalities relative to image-only evidence.
- `sonnet` shows positive but uncertain both-modality gains.
- `opus` does not benefit from both modalities in this run.
- Hi-C-only classification is exactly chance for all three models.

Reliability:

- contradiction detection is poor for all three models
- true strong contradictions are usually predicted as `none`
- action selection mostly defaults to `use_both`
- models rarely choose `use_image`, even when the hidden evaluator action says
  image evidence should be trusted

The key safety result is therefore:

> A terminal-agent workflow did not turn multimodal access into reliable
> biological evidence reconciliation. Some classification gains appear, but the
> agents still fail at knowing when the modalities should not be trusted
> together.

This is consistent with the repository's central framing: multimodal access
should not be treated as an automatic reliability upgrade. It must be evaluated
against clean unimodal controls and explicit contradiction/action metrics.

## Recommended Next Steps

1. Keep the V4 result as a final negative diagnostic rather than tuning prompts
   against it.
2. Add a separate calibrated-exemplar variant only if it is clearly labeled as a
   new setting, not a continuation of this zero-shot final run.
3. Investigate why Hi-C-only terminal-agent predictions are uncorrelated with
   hidden labels despite confident visual rationales.
4. Report contradiction and action accuracy alongside classification in every
   future agent benchmark.
5. Treat `use_both` overuse as a primary safety failure mode, not merely a
   formatting issue.
