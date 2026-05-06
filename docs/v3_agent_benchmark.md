# V3 Claude Agent Benchmark

V3 reframes this repository from training a better fusion classifier to benchmarking frontier agentic reasoning models on multimodal evidence reconciliation.

The first implementation is Claude-first, visual-only, and API-safe by default:

- no real provider calls unless `--real-api --confirm-api-call` are both passed
- mock mode is the default path
- visual tools expose only images, crops, dimensions, and image bytes
- default runs compare `image_only`, `hic_only`, and `both_modalities`
- labels, latent contradiction metadata, reliability proxies, and recommended actions stay evaluator-only
- thinking-enabled Claude calls use `temperature=1.0`, as required by Anthropic

## Final V3 Release Result

The final V3 release comparison uses:

- prompt policy: `evidence_separation`
- tool level: `scientific`
- seeds: `21..40`
- profile: `smoke`, giving `6` samples per seed and `120` samples per complete condition
- models: Claude Haiku 4.5, Claude Opus 4.7, Claude Sonnet 4.6 none, Claude Sonnet 4.6 high

Clean unimodal runs were added after the first final sweep because the
`image_predicted_label` and `hic_predicted_label` fields inside a both-modality
response are diagnostic but not clean unimodal measurements. The release result
therefore compares separately run `image_only`, `hic_only`, and
`both_modalities` conditions on matched seed/sample IDs.

| Model | Image only | Hi-C only | Both modalities | Contradiction acc | Action acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `claude-haiku-4-5 / none` | `60/120 = 0.500` | `66/120 = 0.550` | `60/120 = 0.500` | `64/120 = 0.533` | `64/120 = 0.533` |
| `claude-opus-4-7 / none` | `61/120 = 0.508` | `68/114 = 0.596` | `71/120 = 0.592` | `52/120 = 0.433` | `59/120 = 0.492` |
| `claude-sonnet-4-6 / high` | `63/120 = 0.525` | `67/120 = 0.558` | `69/120 = 0.575` | `56/120 = 0.467` | `64/120 = 0.533` |
| `claude-sonnet-4-6 / none` | `58/120 = 0.483` | `66/120 = 0.550` | `78/120 = 0.650` | `61/120 = 0.508` | `62/120 = 0.517` |

Opus Hi-C-only has `n=114` because seed `40` did not complete before the API
run stopped. It is retained with explicit `n=114`; paired comparisons use common
sample IDs.

Main interpretation:

- Zero-shot image-only classification is near chance.
- Zero-shot Hi-C-only classification has modest signal.
- Sonnet none is the strongest classification condition and the only condition
  with a clear both-modality gain over clean image-only.
- Contradiction and action reasoning remain weak. In the best classification
  condition, true strong contradictions are predicted as `none` in `23/33`
  cases and true weak contradictions in `15/21` cases.
- The final V3 result is diagnostic: models can use some biological priors from
  public evidence, but they do not reliably reconcile conflicting modalities.

Curated release files live under `results/v3/final/`. Raw final roots are:

- `results/v3/final_model_sweep/scientific_20260505T201217Z`
- `results/v3/scientific_tools/sonnet_evidence_sep_20260505T175124Z`
- `results/v3/final_reasoning/sonnet_scientific_high_20260505T203622Z`
- `results/v3/final_unimodal_clean/scientific_20260506T021839Z`

## Smoke Run

```bash
python scripts/run_v3_agent_benchmark.py --profile smoke --mock
```

This generates a six-sample mock run across `cs=0.0`, `cs=0.5`, and `cs=1.0`.
By default it runs all three input conditions. To run only one condition, add
`--input-condition image_only`, `--input-condition hic_only`, or
`--input-condition both_modalities`.

The default prompt policy is `baseline`. To force the agent to judge each
modality separately before reconciling them, add
`--prompt-policy evidence_separation`. Use `--prompt-policy all` for a local
mock check of both policies.

The default tool level is `visual`. To expose fixed non-leaky quantitative
feature tools in addition to visual tools, add:

```bash
--tool-level scientific
```

To jump to Level 3 and let the API model write and run its own Python analysis
over copied public sample files, use:

```bash
--tool-level python
```

## Real API Run

Only run this after confirming API use and setting `ANTHROPIC_API_KEY`:

```bash
python scripts/run_v3_agent_benchmark.py \
  --profile smoke \
  --model claude-sonnet-4-6 \
  --real-api \
  --confirm-api-call
```

Model IDs are CLI-configurable because provider model names can change over
time. The runner tries the short names first, such as `claude-sonnet-4-6`, and
falls back to dated IDs when Anthropic rejects a model as not found.

To compare the planned Claude model set:

```bash
python scripts/run_v3_agent_benchmark.py \
  --profile pilot \
  --model-set claude_high \
  --real-api \
  --confirm-api-call
```

## Prompt-Policy Ablation

The first prompt ablation keeps the dataset, visual tools, model, and thinking
level fixed, and changes only the system prompt:

- `baseline`: compact JSON answer with classification, contradiction status,
  trusted evidence, action, and rationale
- `evidence_separation`: forces `image_evidence`, `hic_evidence`,
  `image_predicted_label`, `hic_predicted_label`, and `modality_agreement`
  before final classification and action

Run only the new policy on the same seeds as the existing baseline:

```bash
RUN_ROOT="results/v3/prompt_ablation/sonnet_evidence_sep_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

for SEED in {21..40}; do
  python scripts/run_v3_agent_benchmark.py \
    --profile smoke \
    --model claude-sonnet-4-6 \
    --input-condition both_modalities \
    --prompt-policy evidence_separation \
    --thinking none \
    --seed "$SEED" \
    --real-api \
    --confirm-api-call \
    --output-dir "$RUN_ROOT/evidence_separation_seed_${SEED}"
done

echo "Wrote runs to: $RUN_ROOT"
```

Then compare it against the saved baseline run:

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/performance/sonnet_assets_20260505T044824Z \
  "$RUN_ROOT" \
  --input-cost-per-mtok 3 \
  --output-cost-per-mtok 15
```

The key readout is whether `none / evidence_separation / both_modalities`
improves contradiction and action metrics, and whether the contradiction
confusion matrix stops collapsing true `weak` and `strong` cases into predicted
`none`.

## Fixed Scientific Tool Run

Level 2 exposes fixed public feature tools:

- `measure_image_features`
- `measure_hic_features`
- `compare_modalities`
- `generate_feature_report`

These tools compute deterministic summaries from public PNG pixels only. They
do not expose hidden labels, latent contradiction metadata, reliability proxies,
or recommended actions.

Run the first Level 2 ablation on the same seeds:

```bash
RUN_ROOT="results/v3/scientific_tools/sonnet_evidence_sep_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

for SEED in {21..40}; do
  python scripts/run_v3_agent_benchmark.py \
    --profile smoke \
    --model claude-sonnet-4-6 \
    --input-condition both_modalities \
    --prompt-policy evidence_separation \
    --tool-level scientific \
    --thinking none \
    --seed "$SEED" \
    --max-workers 3 \
    --real-api \
    --confirm-api-call \
    --output-dir "$RUN_ROOT/scientific_seed_${SEED}"
done

echo "Wrote runs to: $RUN_ROOT"
```

Then compare against visual-only baseline and visual-only evidence separation:

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/performance/sonnet_assets_20260505T044824Z \
  results/v3/prompt_ablation/sonnet_evidence_sep_20260505T145858Z \
  "$RUN_ROOT" \
  --input-cost-per-mtok 3 \
  --output-cost-per-mtok 15
```

## Python Tool Gate

Level 3 adds `run_python_analysis`, which executes model-written Python inside a
temporary per-sample workspace containing only public files for the current input
condition. Hidden manifests, labels, reliability proxies, and source metadata are
not copied into that workspace.

Use a cheap 5-seed gate before any full run:

```bash
RUN_ROOT="results/v3/python_tools/sonnet_gate_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

for SEED in {21..25}; do
  python scripts/run_v3_agent_benchmark.py \
    --profile smoke \
    --model claude-sonnet-4-6 \
    --input-condition both_modalities \
    --prompt-policy evidence_separation \
    --tool-level python \
    --thinking none \
    --seed "$SEED" \
    --max-workers 2 \
    --max-tool-rounds 2 \
    --real-api \
    --confirm-api-call \
    --output-dir "$RUN_ROOT/python_seed_${SEED}"
done
```

Then summarize against previous runs:

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/performance/sonnet_assets_20260505T044824Z \
  results/v3/prompt_ablation/sonnet_evidence_sep_20260505T145858Z \
  results/v3/scientific_tools/sonnet_evidence_sep_20260505T175124Z \
  "$RUN_ROOT" \
  --input-cost-per-mtok 3 \
  --output-cost-per-mtok 15
```

Only scale Level 3 if the gate improves conflict recall without a matching rise
in false conflict calls.

## Final Clean Unimodal Add-On

The final clean unimodal add-on should be run with the same seeds and profile as
the both-modality final sweep. This keeps `(seed, sample_id)` pairings aligned.

```bash
RUN_UNI="results/v3/final_unimodal_clean/scientific_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_UNI"

for SEED in {21..40}; do
  python scripts/run_v3_agent_benchmark.py \
    --profile smoke \
    --seed "$SEED" \
    --model claude-haiku-4-5 \
    --thinking none \
    --prompt-policy evidence_separation \
    --tool-level scientific \
    --input-condition image_only \
    --output-dir "$RUN_UNI/claude-haiku-4-5_image_only_seed_$SEED" \
    --real-api --confirm-api-call

  python scripts/run_v3_agent_benchmark.py \
    --profile smoke \
    --seed "$SEED" \
    --model claude-haiku-4-5 \
    --thinking none \
    --prompt-policy evidence_separation \
    --tool-level scientific \
    --input-condition hic_only \
    --output-dir "$RUN_UNI/claude-haiku-4-5_hic_only_seed_$SEED" \
    --real-api --confirm-api-call
done
```

Repeat the same loop for:

- `claude-sonnet-4-6`, `--thinking none`
- `claude-sonnet-4-6`, `--thinking high`
- `claude-opus-4-7`, `--thinking none`

Summarize the release set with:

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/final_model_sweep/scientific_20260505T201217Z \
  results/v3/scientific_tools/sonnet_evidence_sep_20260505T175124Z \
  results/v3/final_reasoning/sonnet_scientific_high_20260505T203622Z \
  "$RUN_UNI"
```

For faster runs, add bounded sample-level concurrency. Start conservatively:

```bash
--max-workers 3
```

For the six-sample smoke profile this can make each seed substantially faster
because the six sample calls run in parallel. If the provider returns rate-limit
errors, lower the value to `2` or return to the sequential default `1`.

## Current Tool Level

The current benchmark implements Levels 1 through 3 of the tool ladder:

- Level 0: prompt-only visuals
- Level 1: visual-only tools, implemented
- Level 2: fixed non-leaky scientific analysis tools, implemented
- Level 3: sandboxed Python over public image/contact-map files, implemented
- Level 4: open-ended agent workflow

The benchmark should move up this ladder only after the lower level is measured, otherwise it becomes unclear whether gains come from visual reasoning, tool use, leaked metadata, or artifact exploitation.

## Outputs

Each run writes:

- `sample_manifest.jsonl`: evaluator-only manifest with hidden labels
- `predictions.jsonl`: per-sample parsed predictions and scores
- `summary.csv`: model and contradiction-level aggregate metrics
- `report.txt`: compact text report
