# The Radiological Society of North America Presents
 An experiment in detecting intracranel aneuriesm using machine learning and artificial intelligence.

## Research Question

**Primary Question**: Can we develop a machine learning model that accurately detects and classifies brain aneurysms from neuroimaging data (DICOM/NIfTI files) while accounting for patient demographics and brain morphometry?

**Sub-questions**:
- How do age, sex, and brain volume influence aneurysm presentation in neuroimaging?
- Which imaging features are most predictive of aneurysm presence and characteristics?
- Can we identify 13 distinct aneurysm subtypes or characteristics as suggested by the scoring formula?

## Hypothesis

**Primary Hypothesis**: A deep learning model trained on multi-modal neuroimaging data (DICOM/NIfTI) combined with patient demographic features (age, sex, brain volume) will achieve clinically significant performance in:
1. Binary detection of aneurysm presence/absence (AUC_AP)
2. Multi-class classification of 13 aneurysm characteristics (AUC_0 through AUC_12)

**Secondary Hypotheses**:
- H1: Brain volume normalization will improve model generalization across different patient populations
- H2: Age and sex are confounding variables that significantly impact aneurysm morphology
- H3: Ensemble methods combining multiple model architectures will outperform single models

## Experimental Design

### 1. Data Preprocessing Pipeline
```
Raw Data → DICOM/NIfTI Processing → Brain Extraction → Registration → Normalization → Feature Extraction
```

**Key preprocessing steps**:
- Skull stripping using BET or ANTs
- Registration to MNI152 template
- Intensity normalization (Z-score or histogram matching)
- Brain volume calculation
- Aneurysm annotation verification

### 2. Feature Engineering
- **Imaging features**: Voxel intensities, texture features (GLCM, GLRLM), shape descriptors
- **Clinical features**: Age (continuous), Sex (binary), Brain volume (continuous, normalized)
- **Derived features**: Age-adjusted brain volume, vascular territory maps

### 3. Model Architecture Strategy

Given the dual-objective nature (binary + 13-class), propose a multi-task learning approach:

```
Input Layer (3D CNN + Clinical Features)
    ↓
Shared Feature Extractor
    ↓
    ├── Binary Classification Head (AUC_AP)
    └── Multi-class Classification Head (13 outputs: AUC_0...AUC_12)
```

## Testing Approach

### 1. Cross-validation Strategy
- **5-fold stratified cross-validation** ensuring:
  - Balanced aneurysm prevalence across folds
  - Similar age/sex distributions
  - No patient data leakage between folds

### 2. Evaluation Metrics
- **Primary metric**: Competition score = ½(AUC_AP + 1/13 Σ AUC_i)
- **Secondary metrics**: 
  - Sensitivity/Specificity at various thresholds
  - Calibration plots
  - Subgroup analysis by age/sex

### 3. Statistical Testing
- Permutation tests for feature importance
- Bootstrap confidence intervals for AUC scores
- McNemar's test for model comparisons

## Training Strategy

### 1. Progressive Training Approach
```
Phase 1: Train binary classifier (AUC_AP) → Transfer learned features →
Phase 2: Fine-tune for 13-class classification → 
Phase 3: End-to-end multi-task optimization
```

### 2. Data Augmentation
- Spatial: Random rotations (±10°), translations (±5mm)
- Intensity: Gaussian noise, contrast adjustment
- Clinical: SMOTE for minority class oversampling

### 3. Optimization Strategy
- Loss function: Weighted combination of binary and multi-class focal loss
- Learning rate scheduling: Cosine annealing with warm restarts
- Early stopping based on validation competition score

## Learning Methodology

### 1. Iterative Hypothesis Refinement
```
Initial Model → Error Analysis → Hypothesis Update → Model Revision
```

### 2. Ablation Studies
- Impact of each clinical feature
- Contribution of different imaging sequences
- Effect of preprocessing choices

### 3. Interpretability Analysis
- Grad-CAM for visualizing important regions
- SHAP values for clinical feature importance
- Uncertainty quantification using Monte Carlo dropout

## Theory Development Framework

As results accumulate, work toward a unified theory of brain aneurysm detection:

1. **Observational Phase**: Document patterns in successful/failed predictions
2. **Pattern Recognition**: Identify consistent imaging biomarkers
3. **Mechanistic Understanding**: Link findings to vascular pathophysiology
4. **Predictive Framework**: Develop risk stratification model
5. **Clinical Translation**: Validate on external datasets

## Implementation Roadmap

1. **Week 1-2**: Data exploration and preprocessing pipeline
2. **Week 3-4**: Baseline model development
3. **Week 5-6**: Multi-task architecture implementation
4. **Week 7-8**: Hyperparameter optimization
5. **Week 9-10**: Ensemble methods and final optimization
6. **Week 11-12**: Documentation and theory synthesis

This scientific approach treats the competition as a controlled experiment where we systematically test hypotheses about aneurysm detection, building toward a generalizable theory that could advance clinical practice.