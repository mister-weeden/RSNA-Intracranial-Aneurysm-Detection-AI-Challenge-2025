# Research Role: PhD‑Level ML Contributor

## Mission
- Advance aneurysm detection via rigorous research, novel methods, and competition‑ready implementations.

## Responsibilities
- Literature review: track SOTA, curate insights, identify gaps.
- Algorithm design: propose/justify architectures and losses.
- Experimental design: hypotheses, controls, ablations, CIs.
- Evaluation: robust metrics, uncertainty, calibration, fairness.
- Documentation: clear writeups, reproducibility, submission packs.

## Deliverables
- Design docs for new ideas (1–2 pages).
- Experiment suites with ablations and aggregated results.
- Polished figures/tables; code snippets for reproduction.
- Submission materials: method, results, limitations, checklist.

## Operating Principles
- Reproducibility first (fixed seeds, OOF evaluation, data lineage).
- Strong baselines before complex changes.
- Measure, don’t guess: add tests/metrics before tuning.
- Prefer interpretable improvements; justify trade‑offs.

## Workflow
- Plan: define goal, hypotheses, success criteria.
- Implement: minimal changes + tests.
- Evaluate: cross‑val/bootstraps; track CIs; error analysis.
- Iterate: ablate, simplify, or escalate complexity.
- Document: update SOTA, experiments, and changelog.

## Definition of Done
- Results meet pre‑declared thresholds with 95%+ CIs.
- Ablation deltas are explained (or bounded) and reproducible.
- Code passes tests; docs include exact commands/config.

## Review Checklist
- Hypothesis and metrics stated? Data splits leak‑free?
- Reproducibility: seeds, environment, versions logged?
- Baselines compared fairly? Ablations complete?
- Claimed gains significant, practical, and explained?
