# SOTA & Bibliography (Living Document)

## Tracker (high‑level)
- Task: Intracranial aneurysm detection/classification (3D volumes)
- Primary metric: competition score = 0.5·(AUC_AP + mean AUC over 13 traits)
- Related signals: vesselness (Frangi), uncertainty (MC‑dropout), calibration

## Methods By Theme
- 3D CNN backbones: residual/unet/medical‑net variants; multi‑task heads
- Vesselness: Hessian/Frangi multi‑scale filters for vascular enhancement
- Uncertainty: MC‑dropout, test‑time augmentation, ensembling
- Training: focal loss, class balancing, cosine annealing, early stopping
- Evaluation: OOF CV, bootstrap CIs, McNemar, calibration curves

## Papers To Track (fill as we go)
- Aneurysm detection in CTA/MRA with 3D CNNs — venue/year, dataset, score, notes
- Vessel enhancement via Frangi filter — classic reference, parameterization
- MedicalNet/3D transfer learning — source pretraining, adaptation
- MC‑dropout for uncertainty — methodology and best practices
- Calibration in medical AI — ECE, reliability diagrams

## Result Table (examples)
| Paper | Venue/Year | Dataset | Task | Metric | Score | Notes |
|------|------------|---------|------|--------|-------|-------|
| <TBD> | <TBD> | <TBD> | Aneurysm det. | AUC_AP | <TBD> | Baseline |
| <TBD> | <TBD> | <TBD> | 13‑class | mean AUC | <TBD> | Multi‑task |

## Curation Notes
- Prefer external validation and full reporting (splits, CIs).
- Normalize metrics/thresholds across works where possible.
- Log assumptions and preprocessing differences explicitly.

## How To Contribute
- Add entries above with: citation, dataset, splits, metrics, best model, key tricks.
- Keep a short “why it matters” note per paper.
