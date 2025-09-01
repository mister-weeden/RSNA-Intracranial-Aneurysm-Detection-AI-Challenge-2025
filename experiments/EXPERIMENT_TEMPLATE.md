# Experiment Template

## 1. Objective & Hypotheses
- Goal: <what you aim to improve>
- Primary metric: <e.g., competition score, AUC_AP>
- Hypotheses: <H1, H2>

## 2. Dataset & Splits
- Data source: `data/` (version/date)
- Splits: <k‑fold / OOF plan>; no patient leakage
- Preprocessing: <normalize, vesselness, augmentation>

## 3. Model & Config
- Architecture: <e.g., MultiTaskAneurysmNet>
- Key params: {lr: , batch_size: , input_size: }
- Seed: <int> (use `set_deterministic_seed`)

## 4. Ablations
- Components toggled: <vesselness on/off, dropout, uncertainty>
- Rationale: <why each matters>

## 5. Training Plan
- Commands:
  - Run: `python3 demo_complete_pipeline.py` (or training entrypoint)
  - Tests: `python3 -m unittest -v test_suite_comprehensive.py`
- Hardware: <CPU/GPU>

## 6. Metrics & Reporting
- Metrics: ROC AUC, AP, calibration, CIs (bootstrap)
- Plots: ROC/PR, reliability diagram
- Tables: mean ± CI across folds

## 7. Results
- Summary: <numbers>
- Wins/regressions vs baseline: <explain>

## 8. Error Analysis
- Failure modes: <where/why>
- Next fixes: <data, model, loss>

## 9. Reproducibility
- Exact commit: `<hash>`
- Env: `<python/pkgs>`
- Seeds: `<list>`

## 10. Conclusion & Next Steps
- Decision: <adopt/reject>
- Follow‑ups: <experiments to run>
