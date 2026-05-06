# Multimodal Agent Benchmark Plan

This plan separates two related but different research tracks:

- **V3: controlled API benchmark** - we own the dataset, prompt, tools, tool loop, scoring, and leakage controls.
- **V4: terminal-agent workflow benchmark** - Codex, Claude Code, or similar products own more of the agent loop and act like autonomous researchers in a task folder.

The central question is:

> How well can frontier models and agent workflows classify polymer-derived normal/cancer samples and reconcile disagreements between polymer images and Hi-C contact maps?

## Current State

Implemented:

- [x] V3 visual benchmark sample generation from synthetic polymer conformations.
- [x] Image-only, Hi-C-only, and both-modality input conditions.
- [x] Safe evaluator-only hidden labels and sample manifest.
- [x] Claude API runner with explicit `--real-api --confirm-api-call` gate.
- [x] Mock runner for local smoke tests.
- [x] Baseline prompt policy.
- [x] Evidence-separation prompt policy.
- [x] Fixed non-leaky scientific feature tools.
- [x] Sandboxed Python tool gate.
- [x] Summary script that aggregates across seeds, prompt policies, thinking levels, and input conditions.
- [x] Final clean unimodal add-on for image-only and Hi-C-only results.

Final V3 controlled API result on matched seeds `21..40`, evidence-separation prompt, scientific tools:

```text
claude-haiku-4-5 / none / image_only: acc=0.500
claude-haiku-4-5 / none / hic_only: acc=0.550
claude-haiku-4-5 / none / both_modalities: acc=0.500, contradiction_acc=0.533, action_acc=0.533

claude-opus-4-7 / none / image_only: acc=0.508
claude-opus-4-7 / none / hic_only: acc=0.596 (n=114; seed 40 incomplete)
claude-opus-4-7 / none / both_modalities: acc=0.592, contradiction_acc=0.433, action_acc=0.492

claude-sonnet-4-6 / high / image_only: acc=0.525
claude-sonnet-4-6 / high / hic_only: acc=0.558
claude-sonnet-4-6 / high / both_modalities: acc=0.575, contradiction_acc=0.467, action_acc=0.533

claude-sonnet-4-6 / none / image_only: acc=0.483
claude-sonnet-4-6 / none / hic_only: acc=0.550
claude-sonnet-4-6 / none / both_modalities: acc=0.650, contradiction_acc=0.508, action_acc=0.517
```

Interpretation:

- Clean image-only zero-shot classification is near chance for all models.
- Clean Hi-C-only evidence carries modest zero-shot signal, strongest for Opus at `68/114 = 0.596`.
- Sonnet none is the only condition with a clear both-modality classification gain: paired `both - image = +0.167`, approximate 95% interval `[+0.041, +0.292]`; paired `both - hic = +0.100`, interval `[-0.002, +0.202]`.
- Haiku mostly collapses to `normal` and `use_both`; that is a model limitation, not a scoring bug.
- Opus is more sensitive to abnormal/disagreement cues but overcalls cancer and false contradictions.
- Contradiction and action accuracy remain negative diagnostic findings. In the best condition, true strong contradictions are predicted as `none` in `23/33` cases and true weak contradictions in `15/21` cases.

## Core Distinction

### V3 API Benchmark

V3 answers:

> Can the model reconcile modalities when we give it controlled evidence and controlled tools?

We control:

- exact prompt
- exact images and files exposed to the model
- tool schemas
- tool execution
- sandbox limits
- number of tool rounds
- scoring format
- cost and token accounting
- hidden-label isolation

### V4 Terminal-Agent Benchmark

V4 answers:

> Can a full agentic workflow solve the problem as an autonomous researcher?

The terminal agent controls more:

- planning strategy
- file-reading strategy
- shell usage
- Python usage
- intermediate analysis artifacts
- session memory
- default system prompt and product harness behavior
- tool ordering and recovery behavior

This is closer to real use, but less controlled scientifically.

## Flow

```text
Synthetic polymer conformation
        |
        v
Generate public evidence
  - polymer image
  - Hi-C contact map
        |
        v
Evaluator-only metadata
  - true normal/cancer label
  - contradiction status
  - recommended action
  - reliability proxies
        |
        +-----------------------------+
        |                             |
        v                             v
V3 controlled API benchmark       V4 terminal-agent benchmark
  model + controlled tools          Codex / Claude Code + shell/Python
        |                             |
        v                             v
Structured JSON prediction        Structured JSON prediction
        |                             |
        +--------------+--------------+
                       v
             Shared evaluator
        - classification accuracy
        - contradiction detection
        - action accuracy
        - calibration
        - cost and latency
```

## V3 Roadmap: Controlled API Benchmark

### V3 Level 0: Prompt-Only Visual Input

Purpose:

- Measure the base model's visual reasoning without extra tools.
- Establish the lowest controlled capability level.

Work items:

- [ ] Add a mode that sends the panel image only and disables all visual tools.
- [ ] Run mock smoke test.
- [ ] Run real API smoke test on `n=6`.
- [ ] Run stability test with repeated trials on the same assets.
- [ ] Run performance test across different seeds/assets.
- [ ] Compare image-only, Hi-C-only, and both-modality inputs.

Success criteria:

- The run is parseable and reproducible.
- No hidden metadata leaks through filenames, sample IDs, or prompts.
- Baseline classification and contradiction behavior are measured.

### V3 Level 1: Visual Tools

Purpose:

- Let the model inspect the panel, individual modality images, and crops.
- Test whether basic visual tool use helps classification or reconciliation.

Implemented:

- [x] `load_sample_panel`
- [x] `load_modality_image`
- [x] `crop_modality_image`
- [x] input-condition restrictions for unimodal runs
- [x] baseline prompt
- [x] evidence-separation prompt

Test ladder:

- [x] Mock smoke test.
- [x] Real API smoke test.
- [x] Stability test on same assets.
- [x] Performance test across 20 seeds/assets.
- [x] Prompt-policy ablation across 20 seeds/assets.
- [x] Model ablation: Sonnet vs Haiku vs Opus.
- [x] Reasoning ablation: Sonnet none vs Sonnet high on the same assets.

Current bottleneck:

- The model rarely flags true weak/strong contradiction.
- The main failure is reconciliation, not only classification.

Next immediate work:

- [x] Finish `evidence_separation` real API runs across seeds 21-40.
- [x] Summarize against the saved baseline.
- [x] Inspect contradiction confusion matrix.
- [x] Add clean unimodal runs to remove both-modality contamination from self-reported modality labels.
- [ ] Decide whether V4 should keep zero-shot framing or add calibrated exemplar panels as a separate setting.

### V3 Level 2: Fixed Scientific Tools

Purpose:

- Give the model non-leaky quantitative summaries without letting it write arbitrary code.
- Test whether better measurements improve reconciliation while preserving strong experimental control.

Candidate tools:

- [x] `measure_image_features(sample_id)`:
  - compactness
  - radial spread
  - connectedness proxy
  - intensity distribution
- [x] `measure_hic_features(sample_id)`:
  - diagonal strength
  - off-diagonal contact mass
  - compartment-like blockiness
  - distance-decay slope
- [x] `compare_modalities(sample_id)`:
  - non-label structural agreement score
  - feature-level discrepancy summary
  - no hidden labels or reliability proxies
- [x] `generate_feature_report(sample_id)`:
  - JSON-only public quantitative report

Test ladder:

- [x] Unit tests that tools expose no hidden metadata.
- [x] Mock smoke test.
- [x] API smoke test on `n=6`.
- [x] Stability/performance run across 20 seeds/assets.
- [x] Compare Level 1 vs Level 2 on the same seeds.
- [x] Run final model/reasoning/unimodal clean sweep for release summary.

Success criteria:

- Classification can improve without label leakage, but this is not enough.
- Contradiction detection should improve without a matching rise in false conflict calls.
- Action accuracy should improve, especially `use_image` and `use_hic`; final V3 did not meet this criterion.
- Classification gains must be checked against clean unimodal controls, not self-reported modality labels inside both-modality responses.

### V3 Level 3: Sandboxed Python Over Public Files

Purpose:

- Allow the API model to write and run Python analysis over public sample files.
- Test whether open-ended computation helps while we still control the sandbox.

Allowed inputs:

- polymer image PNG
- Hi-C contact map PNG or public NumPy array
- public instructions
- public tool docs

Forbidden inputs:

- `sample_manifest.jsonl`
- true labels
- contradiction labels
- reliability proxies
- source conformation metadata
- train/test labels from the evaluator

Work items:

- [x] Add a `run_python_analysis` tool with a temporary working directory per sample.
- [x] Copy only public sample files into the sandbox.
- [x] Set runtime timeout.
- [x] Set output-size limit.
- [x] Log stdout/stderr through the tool payload.
- [x] Add static safety checks and isolated execution mode.
- [x] Add a final no-tools JSON response after the tool budget is exhausted.
- [x] Add leakage tests.
- [ ] Add cost and latency accounting.

Test ladder:

- [x] Mock smoke test with fake Python output.
- [x] Local sandbox test with harmless scripts.
- [ ] API smoke test on `n=6`.
- [ ] Stability test on same assets.
- [ ] Performance test across 20 seeds/assets.
- [ ] Compare Level 1, Level 2, and Level 3 on identical seeds.

Success criteria:

- The model actually uses Python to measure modality-specific evidence.
- True weak/strong contradictions are flagged more often.
- Recommended actions shift away from always `use_both`.
- Generated code does not access hidden evaluator files.

### V3 Level 4: Controlled Multi-Step API Agent

Purpose:

- Still use the API, but allow a richer controlled workflow.
- This is the bridge between V3 and V4.

Work items:

- [ ] Add explicit planning step.
- [ ] Add separate analysis step for image.
- [ ] Add separate analysis step for Hi-C.
- [ ] Add reconciliation step.
- [ ] Add final JSON-only decision step.
- [ ] Optionally add self-check against schema and visible evidence.

Test ladder:

- [ ] Smoke test.
- [ ] Stability test.
- [ ] Performance test.
- [ ] Compare against Level 3.

Success criteria:

- Better contradiction recognition without losing classification accuracy.
- More interpretable failure modes.
- Less collapse to `contradiction_status=none`.

## V4 Roadmap: Terminal-Agent Workflow Benchmark

### V4 Level 0: Read-Only Task Folder

Purpose:

- Give Codex or Claude Code a clean task folder and ask for JSON output.
- No arbitrary repository context, no hidden metadata.

Task folder shape:

```text
task_001/
  polymer_image.png
  hic_contact_map.png
  instructions.md
  output_schema.json
```

Work items:

- [ ] Create `scripts/build_v4_agent_tasks.py`.
- [ ] Write one task folder per sample.
- [ ] Include only public evidence.
- [ ] Add output schema.
- [ ] Add evaluator that reads terminal-agent outputs.

Agent commands:

```bash
claude -p "$(cat task_001/instructions.md)" \
  --model claude-sonnet-4-6 \
  --output-format json \
  --allowedTools "Read" \
  --permission-mode acceptEdits
```

```bash
codex exec --json \
  --sandbox read-only \
  "$(cat task_001/instructions.md)"
```

Test ladder:

- [ ] One manual task smoke test.
- [ ] Batch smoke test on `n=6`.
- [ ] Stability test on same task folders.
- [ ] Performance test across 20 seeds/assets.

Success criteria:

- Terminal agents return valid JSON.
- No access to hidden evaluator metadata.
- Results are comparable to V3 Level 1.

### V4 Level 1: Read-Only Repository Context

Purpose:

- Let the terminal agent inspect public code/docs that explain the benchmark, but not hidden labels.

Work items:

- [ ] Create a public benchmark package or exported task directory.
- [ ] Add clear instructions and schema.
- [ ] Deny access to `results/`, manifests, and hidden metadata where possible.
- [ ] Run Codex and Claude Code in read-only mode.

Risk:

- The agent may infer unintended patterns from generation code if it sees too much.

Success criteria:

- Better task understanding without leakage.
- No hidden-label access.

### V4 Level 2: Terminal Agent With Python

Purpose:

- Let the terminal agent behave like an analyst: write scripts, compute features, and produce a final decision.

Work items:

- [ ] Provide public task folder.
- [ ] Allow shell/Python in a constrained working directory.
- [ ] Log all commands.
- [ ] Limit runtime and cost.
- [ ] Score final JSON.

Agent commands:

```bash
claude -p "$(cat task_001/instructions.md)" \
  --model claude-sonnet-4-6 \
  --output-format json \
  --allowedTools "Read,Bash" \
  --permission-mode acceptEdits \
  --max-budget-usd 2.00 \
  --max-turns 8
```

```bash
codex exec --json \
  --sandbox workspace-write \
  "$(cat task_001/instructions.md)"
```

Test ladder:

- [ ] Manual one-task run.
- [ ] Mock or dry-run harness.
- [ ] Batch smoke test on `n=6`.
- [ ] Stability test on same assets.
- [ ] Performance test across 20 seeds/assets.
- [ ] Compare against V3 Level 3 API + sandboxed Python.

Success criteria:

- Terminal agents produce useful analysis scripts.
- Contradiction/action metrics improve beyond V3 Level 1.
- Improvements can be attributed to workflow, not hidden metadata.

### V4 Level 3: Autonomous Research Workflow

Purpose:

- Give the agent a higher-level objective and a batch of samples.
- Let it design a strategy, run analyses, and submit predictions.

Work items:

- [ ] Build batch task directories.
- [ ] Add budget and time limits.
- [ ] Add structured final output file requirement.
- [ ] Add replayable logs.
- [ ] Add evaluator for batch predictions.

Success criteria:

- The agent discovers a reusable analysis method.
- Batch performance improves over per-sample ad hoc reasoning.
- Logs are interpretable enough to audit.

## Test Types

### API Call Test

Goal:

- Confirm provider call works, model ID is accepted, output parses, and cost is sane.

Checklist:

- [ ] Use `--profile smoke`.
- [ ] Use `--real-api --confirm-api-call`.
- [ ] Run one model and one prompt policy.
- [ ] Use `--max-workers 2` or `--max-workers 3` only after a sequential smoke run succeeds.
- [ ] Confirm `predictions.jsonl` exists.
- [ ] Confirm `summary.csv` exists.
- [ ] Confirm no malformed outputs or inspect them if present.

### Mock Smoke Test

Goal:

- Verify local code paths without provider cost.

Checklist:

- [ ] Run mock benchmark.
- [ ] Run summarizer.
- [ ] Run tests.
- [ ] Confirm no API key required.

Command:

```bash
python scripts/run_v3_agent_benchmark.py --profile smoke --mock
pytest -q tests/test_v3_agent_benchmark.py
```

### Real Smoke Test

Goal:

- Validate real model behavior on a cheap, tiny run.

Checklist:

- [ ] `n=6` samples.
- [ ] One model.
- [ ] One input condition or all conditions.
- [ ] One prompt policy.
- [ ] Inspect raw outputs for schema problems.

### Stability Test

Goal:

- Measure run-to-run variability on the same assets.

Design:

- Same seed.
- Same samples.
- Same model.
- Same prompt policy.
- Multiple repeated trials.

Interpretation:

- Large variation means single smoke runs are not reliable.
- Stability is about model nondeterminism, not dataset generalization.

### Performance Test

Goal:

- Measure generalization across different polymer conformations/assets.

Design:

- Different seeds.
- Same model/prompt/tools.
- Aggregate across samples and seeds.

Interpretation:

- This is the main result for scientific claims.
- Compare paired seed/sample results whenever possible.

### Prompt-Policy Ablation

Goal:

- Test whether changing only the prompt improves reconciliation.

Current policies:

- `baseline`
- `evidence_separation`

Checklist:

- [ ] Same seeds as baseline.
- [ ] Same model.
- [ ] Same input condition.
- [ ] Same thinking setting.
- [ ] Compare contradiction confusion matrices.
- [ ] Inspect modality-specific fields.

### Tool-Level Ablation

Goal:

- Test whether more capable tools improve reconciliation.

Comparison:

```text
Level 0 prompt-only
Level 1 visual tools
Level 2 fixed scientific tools
Level 3 sandboxed Python
Level 4 controlled multi-step API agent
```

Checklist:

- [ ] Same seeds.
- [ ] Same model.
- [ ] Same prompt policy where possible.
- [ ] Same evaluator.
- [ ] Compare classification, contradiction, action, calibration, cost, and latency.

### Model Ablation

Goal:

- Test whether stronger or different models recognize disagreement better.

Candidates:

- Claude Haiku/Sonnet/Opus.
- OpenAI GPT-5.4-mini/GPT-5.5 or current equivalents.
- Other frontier reasoning models if budget allows.

Checklist:

- [ ] Verify current model IDs before running.
- [ ] Run same seeds and same prompt/tool level.
- [ ] Compare cost-normalized performance.
- [ ] Compare contradiction confusion matrices, not only accuracy.

### Reasoning Ablation

Goal:

- Test whether higher reasoning effort improves reconciliation.

Checklist:

- [ ] Same seeds/assets.
- [ ] Same prompt policy.
- [ ] Same model.
- [ ] Compare `thinking=none` vs `thinking=high`.
- [ ] Track cost and latency.

Current observation:

- High thinking slightly improved classification in one run, but did not clearly improve contradiction/action.

## Metrics

Primary:

- [ ] Classification accuracy.
- [ ] Contradiction accuracy.
- [ ] Binary contradiction F1.
- [ ] Recommended action accuracy.

Critical diagnostics:

- [ ] Contradiction confusion matrix.
- [ ] Action confusion matrix.
- [ ] Prediction rate for `none`, `weak`, `strong`, `unclear`.
- [ ] Performance by intended contradiction strength.
- [ ] Performance by measured true contradiction status.
- [ ] Image-only vs Hi-C-only vs both-modality paired deltas.
- [ ] Cost per sample.
- [ ] Latency per sample.
- [ ] Malformed output rate.

Interpretation warning:

- Contradiction accuracy can look acceptable if many samples have true `none` and the model predicts `none` almost always.
- Always inspect the confusion matrix.

## Immediate Next Actions

### Current Running Work

- [ ] Finish the V3 Level 1 `evidence_separation` prompt-policy ablation.
- [ ] Summarize it with:

```bash
python scripts/summarize_v3_agent_runs.py \
  results/v3/performance/sonnet_assets_20260505T044824Z \
  "$RUN_ROOT" \
  --input-cost-per-mtok 3 \
  --output-cost-per-mtok 15
```

- [ ] Compare against baseline:
  - classification accuracy
  - contradiction accuracy
  - action accuracy
  - true weak/strong -> predicted none collapse
  - `image_predicted_label` vs `hic_predicted_label` disagreement rate

### If Evidence Separation Helps

- [ ] Repeat on a larger sample count.
- [ ] Try high thinking only for evidence-separation both-modality runs.
- [ ] Run model ablation.

### If Evidence Separation Does Not Help

- [x] Move to V3 Level 2 fixed scientific tools.
- [x] Add quantitative non-leaky image and Hi-C feature tools.
- [ ] Test whether measurable features expose disagreement better than visual inspection alone.

### After V3 Level 2/3

- [ ] Start V4 task-folder benchmark.
- [ ] Compare Claude Code and Codex against V3 API models on the same public samples.

## File Map

Current V3 implementation:

- `src/v3_agent_dataset.py` - sample generation and visual asset export.
- `src/v3_visual_tools.py` - visual tools exposed to the model.
- `src/v3_claude_client.py` - Claude API/mock client and prompt policies.
- `src/v3_agent_metrics.py` - output parsing and scoring.
- `scripts/run_v3_agent_benchmark.py` - benchmark runner.
- `scripts/summarize_v3_agent_runs.py` - multi-run summarizer.
- `tests/test_v3_agent_benchmark.py` - V3 tests.
- `docs/v3_agent_benchmark.md` - usage notes.

Planned V3/V4 additions:

- [ ] `src/v3_scientific_tools.py`
- [ ] `src/v3_python_sandbox.py`
- [ ] `scripts/build_v4_agent_tasks.py`
- [ ] `scripts/run_v4_terminal_agent_benchmark.py`
- [ ] `scripts/evaluate_v4_agent_outputs.py`
- [ ] `docs/v4_terminal_agent_benchmark.md`

## Decision Rules

Move from one level to the next only when one of these is true:

- The current level shows a clear bottleneck that the next level directly addresses.
- The current level is stable enough that more runs are unlikely to change the conclusion.
- The next level tests a different scientific question rather than only adding complexity.

Do not claim reconciliation improvement unless:

- true weak/strong contradictions are detected more often,
- action choices shift appropriately away from `use_both`,
- classification does not collapse,
- and the result holds across different seeds/assets.
