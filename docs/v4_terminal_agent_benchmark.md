# V4 Claude Code Terminal-Agent Evaluation

V4 evaluates Claude Code as a terminal agent on clean public task folders. The
goal is measurement: report how much Haiku, Sonnet, and Opus can classify and
reconcile paired polymer-image and Hi-C evidence. Do not tune prompts, samples,
or scoring after seeing model performance.

## Design

The compact final set uses a frozen hidden manifest:

```text
90 total samples
45 normal / 45 cancer

30 true none   = 15 normal / 15 cancer
30 true weak   = 15 normal / 15 cancer
30 true strong = 15 normal / 15 cancer
```

Primary run:

```text
90 samples x 3 models x both_modalities = 270 Claude Code invocations
```

Optional add-on, only if usage remains:

```text
90 samples x 3 models x image_only/hic_only = 540 more invocations
full clean comparison = 810 invocations
```

Use `--effort high` for all three models. If high effort is unavailable for one
model, do not mix effort settings in the final comparison; rerun all models with
the same highest common setting and document the fallback.

## Completed Final Run

The completed compact final run is `v4_compact_20260509T032302Z`. It includes
the primary both-modality run and the clean unimodal add-on:

```text
both_modalities: 270/270 complete
image_only + hic_only: 540/540 complete
total scored rows: 810
malformed outputs: 0
timeouts: 0
nonzero return codes: 0
```

Final aggregate metrics:

| model | both acc | image-only acc | Hi-C-only acc | contradiction acc | action acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| haiku | 0.622 | 0.444 | 0.500 | 0.311 | 0.322 |
| sonnet | 0.567 | 0.478 | 0.500 | 0.333 | 0.378 |
| opus | 0.522 | 0.556 | 0.500 | 0.322 | 0.333 |

Paired classification deltas:

| model | both - image | both - Hi-C |
| --- | ---: | ---: |
| haiku | +0.178 `[+0.044, +0.311]` | +0.122 `[-0.022, +0.267]` |
| sonnet | +0.089 `[-0.034, +0.211]` | +0.067 `[-0.030, +0.164]` |
| opus | -0.033 `[-0.159, +0.092]` | +0.022 `[-0.047, +0.091]` |

The final V4 interpretation is a mixed classification result but a negative
reconciliation result. Haiku shows a clean paired gain over image-only evidence,
but the gain is not consistent across all models or both unimodal controls.
Hi-C-only accuracy is exactly chance for all three models. Contradiction
detection and recommended action selection remain poor, with true strong
contradictions usually predicted as `none`.

Detailed analysis is in `results/v4/final_analysis_report.md`; aggregate
artifacts are in `results/v4/evaluations/v4_compact_20260509T032302Z_full_clean/`.

## Build Public Tasks

Smoke build:

```bash
python scripts/build_v4_agent_tasks.py \
  --profile smoke \
  --run-id v4_smoke_manual \
  --public-root "${TMPDIR:-/tmp}/v4_public_tasks/v4_smoke_manual" \
  --evaluator-root results/v4/evaluator/v4_smoke_manual \
  --conditions all
```

Compact final build:

```bash
RUN_ID="v4_compact_$(date -u +%Y%m%dT%H%M%SZ)"
PUBLIC_ROOT="${TMPDIR:-/tmp}/v4_public_tasks/$RUN_ID"

python scripts/build_v4_agent_tasks.py \
  --profile compact_final \
  --run-id "$RUN_ID" \
  --public-root "$PUBLIC_ROOT" \
  --evaluator-root "results/v4/evaluator/$RUN_ID" \
  --conditions all
```

Each public task folder contains only the visible evidence, instructions,
schema, and an empty `work/` directory. Hidden labels, source indices, latent
metadata, reliability proxies, and evaluator actions live only in
`results/v4/evaluator/<run_id>/hidden_manifest.jsonl`.

The builder writes a contamination audit to:

```text
results/v4/evaluator/<run_id>/contamination_audit.json
```

The build fails if the public task root contains hidden-metadata fragments.

## Dry Run

Mock the full execution path without calling Claude Code:

```bash
python scripts/run_v4_terminal_agent_benchmark.py \
  --task-root "$PUBLIC_ROOT" \
  --output-root "results/v4/runs/${RUN_ID}_mock_both" \
  --models haiku,sonnet,opus \
  --conditions both_modalities \
  --effort high \
  --mock

python scripts/evaluate_v4_agent_outputs.py \
  "results/v4/runs/${RUN_ID}_mock_both" \
  --hidden-manifest "results/v4/evaluator/$RUN_ID/hidden_manifest.jsonl" \
  --output-dir "results/v4/evaluations/${RUN_ID}_mock_both"
```

Preflight real Claude Code with one task per model:

```bash
python scripts/run_v4_terminal_agent_benchmark.py \
  --task-root "$PUBLIC_ROOT" \
  --output-root "results/v4/runs/${RUN_ID}_preflight_both" \
  --models haiku,sonnet,opus \
  --conditions both_modalities \
  --effort high \
  --limit 1
```

Inspect outputs before the final run:

```bash
python scripts/evaluate_v4_agent_outputs.py \
  "results/v4/runs/${RUN_ID}_preflight_both" \
  --hidden-manifest "results/v4/evaluator/$RUN_ID/hidden_manifest.jsonl" \
  --output-dir "results/v4/evaluations/${RUN_ID}_preflight_both"
```

Proceed only if outputs parse, schema failures are understood, and logs do not
show parent-directory or repository access.

## Primary Final Run

Run the primary both-modality condition first:

```bash
python scripts/run_v4_terminal_agent_benchmark.py \
  --task-root "$PUBLIC_ROOT" \
  --output-root "results/v4/runs/${RUN_ID}_both" \
  --models haiku,sonnet,opus \
  --conditions both_modalities \
  --effort high \
  --resume

python scripts/evaluate_v4_agent_outputs.py \
  "results/v4/runs/${RUN_ID}_both" \
  --hidden-manifest "results/v4/evaluator/$RUN_ID/hidden_manifest.jsonl" \
  --output-dir "results/v4/evaluations/${RUN_ID}_both"
```

Report this as: how each model performs when given both modalities and asked to
reconcile them.

For subscription-auth runs, omit `--max-budget-usd`. If a run stops because of
Claude usage limits, the runner detects the usage-limit message, sleeps for
90 minutes, and restarts the pass with completed outputs skipped. The sleep can
be adjusted with `--usage-limit-sleep-seconds`; use
`--no-usage-limit-retry` to restore one-pass behavior. If an explicit API-style
per-call cap is needed, add `--max-budget-usd <amount>`.

## Optional Clean Unimodal Add-On

If usage remains, run clean unimodal controls on the same frozen 90 samples:

```bash
python scripts/run_v4_terminal_agent_benchmark.py \
  --task-root "$PUBLIC_ROOT" \
  --output-root "results/v4/runs/${RUN_ID}_unimodal" \
  --models haiku,sonnet,opus \
  --conditions image_only,hic_only \
  --effort high \
  --resume

python scripts/evaluate_v4_agent_outputs.py \
  "results/v4/runs/${RUN_ID}_both" \
  "results/v4/runs/${RUN_ID}_unimodal" \
  --hidden-manifest "results/v4/evaluator/$RUN_ID/hidden_manifest.jsonl" \
  --output-dir "results/v4/evaluations/${RUN_ID}_full_clean"
```

Only after this add-on completes should the report discuss paired deltas:

```text
both_modalities - image_only
both_modalities - hic_only
hic_only - image_only
```

## Metrics

Primary both-modality metrics:

- classification accuracy
- contradiction accuracy
- binary contradiction F1
- recommended action accuracy
- contradiction confusion matrix
- action confusion matrix
- malformed output rate
- timeout rate
- cost and latency per model

With `n=90`, interpret only large effects strongly. For a single model's
classification accuracy against chance, about `55/90 = 0.611` is the rough
threshold for `p < 0.05`.

## Reporting Rule

Do not claim multimodal improvement if only the primary both-modality run
completed. In that case, V4 reports each model's direct reconciliation
performance only. Multimodal improvement claims require clean `image_only` and
`hic_only` runs on the same frozen sample IDs.
