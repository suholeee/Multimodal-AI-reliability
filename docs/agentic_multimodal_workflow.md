# Agentic Workflow for Multimodal Evidence Reconciliation

This document links the repository to an agentic workflow for Codex, Claude Code, or similar coding agents. It is designed around the current state of the project: a synthetic image/genomic reliability benchmark where naive fusion did not reliably reconcile contradiction.

## Current Release State

The release now contains two complementary negative/diagnostic tracks:

- V2 supervised PyTorch models: learned fusion did not reliably beat the strongest unimodal or policy baselines once contradiction observability became central.
- V3 zero-shot agents: Sonnet none reached the strongest classification result at `78/120 = 0.650` with scientific tools and both modalities, but contradiction accuracy was only `61/120 = 0.508` and action accuracy was `62/120 = 0.517`.

For V3, use `results/v3/final/README.md` as the release-facing result. Raw smoke, stability, prompt-ablation, and Python-tool gate directories are provenance, not the headline result.

## Target Capability

The project should move from "fusion classifier" toward "evidence reconciliation system." A useful system should output:

- task prediction
- calibrated uncertainty
- modality-specific reliability estimates
- disagreement/contradiction type
- action: fuse, trust image, trust genomics, abstain, or request more evidence

## Recommended Agent Roles

Use separate agent sessions or prompts for distinct work products.

### Scientist

Goal: define one falsifiable reconciliation hypothesis.

Prompt:

```text
Read README.md, results/reports/final_results_summary.txt, and results/v2/reports/*.txt. Propose one falsifiable hypothesis for improving multimodal reconciliation. Specify the expected movement in confident contradiction rate, oracle regret, calibration, and selective accuracy.
```

### Regime Engineer

Goal: improve the benchmark so disagreement is observable and labeled.

Prompt:

```text
Modify only the synthetic regime/data layer. Add explicit latent metadata for modality reliability, artifact/noise state, and contradiction type. Keep labels balanced and preserve train/val/test separation. Add a debug validation check that both unimodal branches retain usable signal while confident contradiction is nonzero.
```

### Model Engineer

Goal: add a reconciliation architecture without erasing baselines.

Prompt:

```text
Add one model or policy that predicts task output plus modality reliabilities. Compare it with image-only, genomic-only, late fusion, learned fusion, genomic-first, confidence routing, oracle branch, and abstention. Do not remove existing baselines.
```

### Skeptic

Goal: find ways the apparent improvement could be fake.

Prompt:

```text
Review the latest change for leakage, validation/test contamination, overfitting to synthetic artifacts, branch collapse, or metrics that hide disagreement. Prioritize concrete file/line findings and missing tests.
```

### Reproducer

Goal: rerun and summarize.

Prompt:

```text
Run the debug V2 sweep. If it passes, run the default V2 sweep. Summarize the exact command, generated reports, validation gate outcome, strongest-regime accuracy, confident contradiction rate, oracle regret, and selective accuracy.
```

## High-Value Project Improvements

### 1. Make Contradiction Ground Truth Explicit

The current V2 regime exposes structural contradiction metadata, but the final observability pass still produced zero confident prediction contradiction. Add explicit labels for:

- no conflict: both modalities support the same latent state
- complementary evidence: each modality sees a different but compatible factor
- coherent biological conflict: local/image evidence and distal/genomic evidence point in different directions
- artifact conflict: one modality is corrupted or unreliable
- missing/low-quality modality

This lets the model be evaluated on reconciliation, not just final classification.

### 2. Separate Biology From Assay Reliability

Represent each sample as:

```text
z_shared: underlying biological state
z_image_private: image-visible biology
z_genomic_private: genomic-visible biology
r_image: image reliability/quality
r_genomic: genomic reliability/quality
c_type: contradiction type
y: target label
```

Then require the model to infer both `y` and the reliability/action decision.

### 3. Add Reconciliation Metrics

Add metrics that directly answer whether the model reconciles modalities:

- conflict detection AUROC/AUPRC against latent `c_type`
- reliability-label accuracy for `r_image` and `r_genomic`
- oracle-branch regret in disagreement strata
- action accuracy: image/genomic/fuse/abstain
- calibration error split by agreement vs disagreement
- risk-coverage curves for selective prediction
- counterfactual consistency when one modality is replaced, corrupted, or masked

### 4. Add Reconciliation Models

Good next candidates:

- shared/private encoders with an explicit disagreement head
- mixture-of-experts gate over image, genomic, fusion, and abstain actions
- product-of-experts for agreement plus mixture-of-experts for conflict
- modality dropout to reduce collapse to one branch
- temperature scaling or conformal abstention for calibrated decisions
- auxiliary losses for branch reliability, contradiction type, and reconstruction

### 5. Add Real Multimodal Datasets Later

Keep the synthetic benchmark as the controlled sandbox. Then add a real-data adapter layer for paired biological assays such as microscopy/contact-map pairs, single-cell multiome, CITE-seq, or 4DN-style chromatin datasets. The adapter should produce the same schema as the synthetic dataset so reconciliation metrics transfer.

## Minimal Implementation Roadmap

1. Add tests for `v2_metrics.py`, `v2_policies.py`, and synthetic split invariants.
2. Extend `src/v2_regime.py` with explicit `latent_reliability_image`, `latent_reliability_genomic`, and `latent_contradiction_type`.
3. Add a reconciliation policy in `src/v2_policies.py` that can output image, genomic, fusion, or abstain.
4. Add report columns for conflict detection, reliability accuracy, action accuracy, and agreement-stratified calibration.
5. Run `python scripts/run_v2_phase_diagram.py --profile debug --no-show`.
6. If promising, run `python scripts/run_v2_phase_diagram.py --profile default --no-show`.
7. Only after synthetic evidence is stable, add a real-data adapter and provenance document.

## Pull Request Template for Agents

```text
Hypothesis:

Files changed:

Commands run:

Validation gate:

Strongest-regime metrics:
- image accuracy:
- genomic accuracy:
- learned fusion accuracy:
- best policy accuracy:
- confident contradiction rate @0.70:
- oracle regret:
- selective accuracy:

Interpretation:

Known limitations:
```
