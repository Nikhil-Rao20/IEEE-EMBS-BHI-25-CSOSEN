<div align="center">

<table border="0">
<tr>
<td width="200" align="center">
<img src="imgs/minder.png" alt="MINDER Logo" width="160"/>
</td>
<td>
<h2>
Machine Learning Framework for Depression Score Analysis in Mindfulness Interventions across Medically Complex Patients
</h2>
</td>
</tr>
</table>

</div>

<div align="center">

**Nikhileswara Rao Sulake¹† · Sai Manikanta Eswar Machara¹ · Divya Katam²**

<sub>¹ Department of Computer Science and Engineering, RGUKT, Nuzvid, India</sub>  <br>
<sub>² Department of Electronics and Communication Engineering, RGUKT, Nuzvid, India</sub><br>
<sub>† Team Leader · 📧 nikhil01446@gmail.com · 🌐 [nikhil-rao20.github.io](https://nikhil-rao20.github.io/) </sub>
</div>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"></a>
  <a href="https://bhi.embs.org/2025/"><img src="https://img.shields.io/badge/IEEE-BHI%202025-red.svg" alt="Conference"></a>
  <a href="Reports/final-submission.tex"><img src="https://img.shields.io/badge/Paper-LaTeX-green.svg" alt="Paper"></a>
</p>

**IEEE EMBS BHI 2025 · Track-1 Data Competition · Team CSOSEN**

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#key">Key Findings</a> •
  <a href="#">Repository Structure</a> •
  <a href="#">Methodology Overview</a> •
  <a href="#">Results</a> •
  <a href="#">Reproducibility</a> •
  <a href="#">Usage</a> •
</p>


---

## 📖 Abstract
<p id='abstract'> </p>
Depression prediction in medically complex populations remains challenging due to heterogeneous treatment responses. We present a comprehensive machine learning framework evaluating **40 models across five methodological phases** to predict Beck Depression Inventory-II (BDI-II) scores at 12- and 24-week follow-ups post-mindfulness intervention. Using data from **210 patients** with diverse medical comorbidities, Transformer and CatBoost models achieved optimal performance (R² = 0.247 and 0.200, respectively). 

Disease-stratified analysis reveals profound condition-dependent effects: cancer patients show elevated depression (+2.92 points) yet strongest therapy benefits (4.19-point improvement with high engagement), while renal patients exhibit unexpected protective patterns (–4.23 points). SHAP analysis identifies baseline severity (≈40%), age (≈15%), and therapy engagement (≈12%) as primary predictors. Disease-specific models achieve exceptional accuracy (R² = 0.81–0.93), establishing condition-stratified frameworks as essential for clinical deployment in precision psychiatry.

In addition, we implemented rigorous statistical validation using 10,000-iteration bootstrap confidence intervals and Mann-Whitney U tests with effect sizes to address small sample concerns. We performed detailed phase-level and model-level visualizations (radar plots, heatmaps), quantified computational efficiency and hardware requirements, and provided translational guidance for clinical deployment.

**📊 Full results, figures, and reproducibility materials are available in this repository.**

---

## 🎯 Key Findings
<p id='key'>
</p>
### 1. **Timepoint-Specific Model Performance**

| Timepoint | Best Model | R² Score | MAE | RMSE | Clinical Utility |
|-----------|-----------|----------|-----|------|-----------------|
| **12 Weeks** | Transformer | 0.247 ± 0.089 | 4.53 ± 0.56 | 5.97 ± 0.78 | ✅ 91% within ±5 points |
| **24 Weeks** | CatBoost | 0.200 ± 0.127 | 4.36 ± 0.58 | 6.14 ± 0.89 | ✅ 88% within ±5 points |

### 2. **Disease-Stratified Models: Exceptional Performance**

| Medical Condition | Sample Size | R² Score | Improvement vs General Model |
|-------------------|-------------|----------|------------------------------|
| Cancer | n=108 | **0.928** | +276% |
| Acute Coronary Syndrome | n=39 | **0.812** | +228% |
| Renal Insufficiency | n=10 | **0.922** | +273% |
| Lower-Limb Amputation | n=10 | **0.888** | +260% |

*General population model: R² = 0.247*

### 3. **Condition-Dependent Treatment Effects**

| Condition | Baseline Depression | Therapy Completion | Engagement-Outcome Correlation | Effect Size |
|-----------|-------------------|-------------------|-------------------------------|-------------|
| **Cancer** | +2.92 points (p=0.007) | 77.2% (↑27.4%) | r = -0.232* (p=0.016) | **-4.19 points** (High vs Low) |
| **Renal** | -4.23 points (p=0.017) | 58.7% | r = +0.656* (p=0.039) | +3.40 points |
| **ACS** | -2.95 points (p=0.044) | 49.1% (↓24.0%) | ρ = +0.374* (p=0.019) | +2.83 points |
| **Amputation** | +1.09 points (ns) | 43.8% (↓25.3%) | r = +0.333 (ns) | +3.80 points |

*\*Statistically significant (p < 0.05)*

### 4. **Feature Importance Hierarchy**

```
🥇 Baseline BDI-II Score    ~40% of variance
🥈 Age                       ~15% of variance
🥉 Therapy Engagement        ~12% of variance (↑296% from 12W to 24W)
   Medical Conditions        ~15% combined
   Demographics              ~10%
   Other Factors             ~8%
```

### 5. **Statistical Validation (Bootstrap + Mann-Whitney U Tests)**

| Comparison | Timepoint | p-value | Cohen's d | Interpretation |
|------------|-----------|---------|-----------|----------------|
| Phase 2 vs Phase 5 (MAE) | 12W | **0.006** | 1.12 | Large effect |
| Phase 3 vs Phase 1 (R²) | 24W | **0.002** | 1.24 | Large effect |
| Phase 3 vs Phase 5 (R²) | 24W | **0.006** | 0.95 | Large effect |

*Based on 10,000-iteration bootstrap resampling*

### 6. **Computational Efficiency: CPU vs GPU**

| Phase | Hardware | Training Time | Performance (R²) | Efficiency Score |
|-------|----------|---------------|------------------|------------------|
| **Phase 2-3** (Classical/Ensemble) | CPU (16GB RAM) | 30 min - 2 hrs | 0.10-0.20 | ⭐⭐⭐⭐⭐ |
| **Phase 4-5** (Deep Learning) | GPU (RTX 3060+) | 2-8 hrs | 0.12-0.25 | ⭐⭐⭐ |

**Key Insight:** Classical/ensemble models achieve **90-95% of deep learning performance** at **10-50× lower computational cost**.

---

## 📁 Repository Structure

```
IEEE_EMBS_BHI_25_CSOSEN/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 LICENSE                            # MIT License
│
├── 📂 Track1_Data/                       # Competition dataset
│   ├── train.csv                         # Training data (n=210)
│   ├── test.csv                          # Test data
│   └── data_dictionary.pdf               # Feature descriptions
│
├── 📂 SRC_Track1/                        # Source code modules
│   ├── data_loader.py                    # Data loading utilities
│   ├── preprocessing.py                  # Feature engineering pipeline
│   ├── models/                           # Model implementations
│   │   ├── phase1_linear.py              # Linear baselines
│   │   ├── phase2_classical_ml.py        # Random Forest, SVR, KNN, etc.
│   │   ├── phase3_ensembles.py           # XGBoost, CatBoost, Stacking
│   │   ├── phase4_deep_learning.py       # MLP, Attention, ResNet
│   │   └── phase5_timeseries.py          # Transformer, LSTM, GRU
│   ├── evaluation.py                     # Cross-validation & metrics
│   ├── interpretability.py               # SHAP analysis
│   └── statistical_tests.py              # Bootstrap CI, Mann-Whitney U
│
├── 📂 Notebooks/                         # Jupyter notebooks
│   ├── 01_EDA.ipynb                      # Exploratory data analysis
│   ├── 02_Feature_Engineering.ipynb      # Feature creation
│   ├── 03_Phase1_Linear_Models.ipynb     # Phase 1 experiments
│   ├── 04_Phase2_Classical_ML.ipynb      # Phase 2 experiments
│   ├── 05_Phase3_Ensembles.ipynb         # Phase 3 experiments
│   ├── 06_Phase4_Deep_Learning.ipynb     # Phase 4 experiments
│   ├── 07_Phase5_Time_Series.ipynb       # Phase 5 experiments
│   ├── 08_Statistical_Validation.ipynb   # Bootstrap & significance tests
│   ├── 09_Disease_Stratified.ipynb       # Condition-specific models
│   └── 10_Interpretability_SHAP.ipynb    # SHAP analysis
│
├── 📂 Results_12W/                       # 12-week prediction results
│   ├── Conference_Submission/
│   │   ├── all_results_compiled.json     # Comprehensive results (5-fold CV)
│   │   ├── phase_performance.csv         # Phase-level aggregates
│   │   └── best_models_summary.json      # Top performers
│   └── Disease_Specific/                 # Condition-stratified results
│
├── 📂 Results_24W/                       # 24-week prediction results
│   ├── Conference_Submission/
│   │   ├── all_results_compiled.json
│   │   ├── phase_performance.csv
│   │   └── best_models_summary.json
│   └── Disease_Specific/
│
├── 📂 Reports/                           # Documentation & visualizations
│   ├── final-submission.tex              # Full manuscript (LaTeX)
│   ├── final-submission.pdf              # Compiled PDF
│   ├── MANUSCRIPT_UPDATES_SUMMARY.md     # Change log
│   ├── FINAL_INTEGRATION_COMPLETE.md     # Integration summary
│   ├── QUICK_REFERENCE.md                # Quick facts & compile steps
│   │
│   └── 📂 figures/                       # All visualizations (publication-ready)
│       ├── phase_radar_12w_actual.png
│       ├── phase_radar_24w_actual.png
│       ├── phase_models_radar_12w_detailed.png
│       ├── phase_models_radar_24w_detailed.png
│       ├── bootstrap_confidence_intervals.png
│       ├── computational_efficiency_analysis.png
│       ├── shap_feature_importance_static.png
│       ├── condition_analysis_overview.png
│       ├── treatment_response_comprehensive_analysis_static.png
│       ├── phase_comparison_heatmap_actual.png
│       ├── phase_comparison_barchart_actual.png
│       └── [30+ additional figures & tables]
│
├── 📂 All_Trained_Models/                # Serialized model artifacts
│   ├── all_models_summary.json           # Model registry
│   ├── Rank_01_phase1_lasso_regression/
│   ├── Rank_01_phase3_catboost/
│   ├── Rank_01_phase5_transformer/
│   └── [40+ model directories with hyperparameters & weights]
│
├── 📓 sample.ipynb                       # Main analysis notebook
└── 📓 Testing.ipynb                      # Model testing & validation
```

---

## 🔬 Methodology Overview

### Five-Phase Modeling Framework

We systematically evaluated **40 models** across five methodological phases to identify optimal architectures for depression outcome prediction:

<div align="center">

| Phase | Category | Key Models | # Models | Best 12W R² | Best 24W R² |
|-------|----------|------------|----------|-------------|-------------|
| **Phase 1** | Linear Baselines | Lasso, Ridge, ElasticNet, Bayesian Ridge | 8 | 0.178 | 0.164 |
| **Phase 2** | Classical ML | Random Forest, SVR, KNN, Gradient Boosting | 10 | 0.089 | 0.138 |
| **Phase 3** | Ensembles | XGBoost, CatBoost, Stacking, Voting | 5 | 0.166 | **0.200** |
| **Phase 4** | Deep Learning | MLP variants, Attention, ResNet-inspired | 7 | 0.183 | 0.162 |
| **Phase 5** | Time-Series | Transformer, LSTM, GRU, ARIMA | 10 | **0.247** | 0.143 |

</div>

### Evaluation Framework

- **Cross-Validation:** 5-fold stratified CV (preserving severity distributions)
- **Metrics:** R² Score, MAE, RMSE
- **Statistical Validation:**
  - Bootstrap confidence intervals (10,000 iterations)
  - Mann-Whitney U tests (non-parametric)
  - Cohen's d effect sizes
- **Interpretability:** SHAP (SHapley Additive exPlanations)
- **Hardware:** Acer Nitro 5 (Intel i7-12650H, 16GB RAM, no GPU)

### Feature Engineering (26 Features)

1. **Demographics:** Age, sex, age², age categories
2. **Clinical Baseline:** BDI-II score, log(BDI-II), BDI-II², severity categories
3. **Medical Comorbidities:** One-hot encoded conditions + subtypes, disease burden indices
4. **Therapy Engagement:** Completion rate, sessions started/completed, adherence levels
5. **Interactions:** Baseline×age, disease burden×engagement, etc.

---

## 📊 Results Summary

### Comprehensive 40-Model Performance Comparison

The following table presents all 40 models evaluated across both 12-week and 24-week prediction timepoints. Models are organized by phase (increasing complexity from Phase 1 to Phase 5), with top performers highlighted for each timepoint.

<div align="center">

#### Table 1: Complete Model Performance Metrics (All 40 Models)

| Model | 12-Week R² | 12-Week MAE | 24-Week R² | 24-Week MAE | Model | 12-Week R² | 12-Week MAE | 24-Week R² | 24-Week MAE |
|-------|------------|-------------|------------|-------------|-------|------------|-------------|------------|-------------|
| **PHASE 1: Linear Baselines** | | | | | **PHASE 3: Advanced Ensembles** | | | | |
| Linear Regression | -0.03 ± 0.24 | 5.18 ± 0.34 | -0.02 ± 0.25 | 5.08 ± 0.65 | Voting Regressor | -0.17 ± 0.35 | 5.39 ± 0.55 | 0.09 ± 0.30 | 4.70 ± 0.42 |
| Ridge Regression | 0.15 ± 0.17 | 4.79 ± 0.38 | 0.09 ± 0.21 | 4.73 ± 0.38 | Stacking Regressor | 0.07 ± 0.12 | 5.18 ± 0.72 | 0.11 ± 0.19 | 4.80 ± 0.48 |
| **Lasso Regression** ⭐ | **0.18 ± 0.11** | **4.71 ± 0.53** | 0.09 ± 0.18 | 4.68 ± 0.43 | Advanced Stacking | 0.04 ± 0.16 | 5.26 ± 0.66 | 0.13 ± 0.18 | 4.77 ± 0.46 |
| **Elastic Net** ⭐ | **0.16 ± 0.16** | **4.77 ± 0.43** | 0.09 ± 0.19 | 4.68 ± 0.41 | XGBoost | 0.09 ± 0.14 | 5.00 ± 0.63 | 0.13 ± 0.24 | 4.63 ± 0.55 |
| Bayesian Ridge | 0.14 ± 0.17 | 4.79 ± 0.36 | 0.08 ± 0.21 | 4.74 ± 0.39 | **CatBoost** 🥇 | 0.09 ± 0.16 | 4.91 ± 0.51 | **0.20 ± 0.13** | **4.36 ± 0.58** |
| Huber Regressor | 0.11 ± 0.16 | 4.79 ± 0.49 | 0.02 ± 0.20 | 4.68 ± 0.21 | | | | | |
| RANSAC Regressor | -0.37 ± 0.39 | 5.98 ± 1.05 | -0.96 ± 0.76 | 6.80 ± 1.49 | **PHASE 4: Deep Learning** | | | | |
| Decision Tree | -0.07 ± 0.14 | 5.19 ± 0.64 | -0.06 ± 0.40 | 4.81 ± 0.63 | MLP (Small) | 0.05 ± 0.09 | 5.24 ± 0.65 | 0.06 ± 0.14 | 4.88 ± 0.08 |
| | | | | | MLP (Medium) | 0.13 ± 0.13 | 4.96 ± 0.31 | 0.15 ± 0.15 | 4.73 ± 0.17 |
| **PHASE 2: Classical ML** | | | | | **MLP (Large)** ⭐ | -0.04 ± 0.18 | 5.30 ± 0.98 | **0.16 ± 0.13** | **4.64 ± 0.09** |
| Random Forest | 0.09 ± 0.19 | 4.82 ± 0.45 | 0.14 ± 0.25 | 4.52 ± 0.44 | TF MLP (Simple) | -0.12 ± 0.16 | 5.29 ± 0.85 | 0.04 ± 0.06 | 4.66 ± 0.12 |
| Extra Trees | 0.05 ± 0.25 | 4.93 ± 0.36 | -0.01 ± 0.39 | 4.78 ± 0.38 | TF MLP (Deep) | -0.35 ± 0.70 | 5.53 ± 1.27 | -0.31 ± 0.32 | 5.49 ± 0.93 |
| AdaBoost | 0.09 ± 0.15 | 4.83 ± 0.53 | 0.11 ± 0.23 | 4.65 ± 0.53 | TF ResNet | 0.05 ± 0.11 | 4.83 ± 0.66 | -0.31 ± 0.27 | 5.31 ± 0.72 |
| Gradient Boosting | 0.06 ± 0.19 | 5.07 ± 0.48 | 0.03 ± 0.26 | 4.79 ± 0.54 | TF Attention | 0.15 ± 0.07 | 4.78 ± 0.34 | 0.12 ± 0.07 | 4.62 ± 0.08 |
| SVR (Linear) | 0.14 ± 0.12 | 4.67 ± 0.58 | 0.03 ± 0.15 | 4.56 ± 0.26 | | | | | |
| SVR (RBF) | 0.15 ± 0.08 | 4.73 ± 0.81 | 0.11 ± 0.18 | 4.49 ± 0.34 | **PHASE 5: Time-Series Models** | | | | |
| SVR (Poly) | 0.04 ± 0.22 | 5.24 ± 1.06 | -0.01 ± 0.10 | 4.80 ± 0.69 | ARIMA | -0.21 ± 0.21 | 5.71 ± 0.97 | -0.23 ± 0.25 | 5.14 ± 1.27 |
| NU SVR | 0.14 ± 0.07 | 4.74 ± 0.91 | 0.05 ± 0.10 | 4.51 ± 0.56 | Exponential Smoothing | -0.20 ± 0.06 | 5.87 ± 0.83 | -0.29 ± 0.38 | 5.31 ± 1.53 |
| KNN Regressor | 0.13 ± 0.14 | 5.01 ± 0.57 | 0.12 ± 0.19 | 4.55 ± 0.56 | Moving Average | -0.50 ± 0.35 | 6.38 ± 1.08 | -0.40 ± 0.23 | 5.80 ± 1.39 |
| KNN Uniform | 0.15 ± 0.14 | 4.95 ± 0.64 | 0.01 ± 0.22 | 4.62 ± 0.67 | LSTM (Simple) | 0.05 ± 0.06 | 5.33 ± 0.48 | 0.01 ± 0.03 | 5.08 ± 0.34 |
| | | | | | LSTM (Bi-dir) | 0.13 ± 0.07 | 5.13 ± 0.57 | -0.01 ± 0.03 | 5.19 ± 0.34 |
| | | | | | LSTM (Stacked) | 0.13 ± 0.12 | 5.03 ± 0.61 | -0.01 ± 0.01 | 5.11 ± 0.37 |
| | | | | | GRU | 0.08 ± 0.05 | 5.22 ± 0.47 | 0.01 ± 0.04 | 5.18 ± 0.37 |
| | | | | | **Transformer** 🥇 | **0.24 ± 0.09** | **4.53 ± 0.56** | 0.08 ± 0.14 | 4.97 ± 0.61 |
| | | | | | **RF Trajectory** ⭐ | 0.01 ± 0.20 | 5.12 ± 0.02 | **0.16 ± 0.12** | **4.80 ± 0.28** |
| | | | | | Ridge Trajectory | 0.07 ± 0.15 | 4.99 ± 0.23 | 0.10 ± 0.15 | 4.95 ± 0.38 |

**Legend:** 🥇 = Best Overall | ⭐ = Top 3 for respective timepoint

</div>

**Key Observations:**
- **12-Week Champion:** Transformer (Phase 5) achieves R² = 0.247, leveraging attention mechanisms for short-term prediction
- **24-Week Champion:** CatBoost (Phase 3) achieves R² = 0.200, with gradient boosting excelling for longer horizons
- **Consistent Performers:** Lasso/ElasticNet (Phase 1) and MLP Large (Phase 4) show balanced performance
- **Phase Trends:** Linear models provide strong baselines; ensembles dominate 24W; time-series models show high variance

---

### BDI-II Trajectory Distributions by Condition

<div align="center">

![BDI Trajectories 1](imgs/temp1.png)
![BDI Trajectories 2](imgs/temp2.png)

**Figure 1:** Longitudinal depression trajectories across different medical conditions, illustrating heterogeneous treatment responses and baseline severity patterns.

</div>

### Phase-Level Performance Comparison

<div align="center">

<img src="imgs/phase_radar_12w_actual.png" alt="Phase Radar 12W" width="45%"/> <img src="imgs/phase_radar_24w_actual.png" alt="Phase Radar 24W" width="45%"/>

**Figure 2:** Phase-level performance radar plots. Left: 12-week predictions. Right: 24-week predictions. Phase 2 (Classical ML) excels at short-term, while Phase 3 (Ensembles) dominates long-term forecasting.

</div>

### Detailed Model-Level Performance (All 40 Models)

<div align="center">

![Detailed 12W](imgs/phase_models_radar_12w_detailed.png)
![Detailed 24W](imgs/phase_models_radar_24w_detailed.png)

**Figure 3:** Individual model performance within each phase. Reveals within-phase heterogeneity and robust algorithmic choices.

</div>

### Hyperparameter Optimization Analysis

<div align="center">

![Hyperparameter Tuning](imgs/hyper-parameter-comp.png)

**Figure 4:** Hyperparameter tuning heatmaps for top-performing models: Transformer (left, 12-week task) and CatBoost (right, 24-week task). Color intensity represents R² score magnitude. White stars mark optimal configurations discovered through Bayesian optimization.

</div>

### SHAP Feature Importance Analysis

<div align="center">

<img src="imgs/feature_importance.png" alt="SHAP Feature Importance" width="60%"/>

**Figure 5:** Global SHAP feature importance rankings. Baseline BDI-II dominates (~40%), followed by age (~15%) and therapy engagement (~12%).

</div>

### Disease-Specific Performance Analysis

<div align="center">

![Disease Analysis](imgs/disease_specific_analysis.png)

**Figure 6:** Comprehensive disease-specific analysis showing condition-dependent treatment effects, engagement patterns, and outcome distributions across cancer, ACS, renal, and amputation patient subgroups.

</div>

---

## 🎨 Visualizations

All figures are publication-ready (300 DPI) and available in [`imgs/`](imgs/):

| Category | Files |
|----------|-------|
| **BDI Trajectories** | `temp1.png`, `temp2.png` |
| **Hyperparameter Optimization** | `hyper-parameter-comp.png` |
| **Phase Performance Radars** | `phase_radar_12w_actual.png`, `phase_radar_24w_actual.png` |
| **Detailed Model Radars** | `phase_models_radar_12w_detailed.png`, `phase_models_radar_24w_detailed.png` |
| **Feature Importance** | `feature_importance.png` |
| **Disease-Specific Analysis** | `disease_specific_analysis.png` |

**Additional resources** including statistical test results and computational profiles are available in [`Reports/figures/`](Reports/figures/).

---

## 🔄 Reproducibility

### ⚠️ Important Note

**All experimental results are comprehensively documented in this repository.** To save computational resources and time, we recommend reviewing the pre-computed results rather than re-running the entire pipeline:

- ✅ **Pre-computed Results:** [`Results_12W/`](Results_12W/) & [`Results_24W/`](Results_24W/) contain all 5-fold CV results
- ✅ **Figures & Tables:** [`Reports/figures/`](Reports/figures/) has 30+ publication-ready visualizations
- ✅ **Trained Models:** [`All_Trained_Models/`](All_Trained_Models/) stores 40+ serialized models with hyperparameters
- ✅ **Statistical Tests:** CSV exports with full bootstrap & Mann-Whitney U results

### If You Still Want to Replicate

<details>
<summary><b>Click to expand replication instructions</b></summary>

#### Hardware Requirements

**Minimum (Phases 1-3):**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- Storage: 10GB
- Training Time: ~6 hours

**Recommended (All Phases):**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 32GB
- GPU: NVIDIA RTX 3060+ (12GB VRAM) for Phases 4-5
- Storage: 20GB
- Training Time: ~36 hours

#### Software Environment

```bash
# Python 3.8+
Python 3.8.10

# Core Libraries
scikit-learn==1.2.2
xgboost==1.7.5
catboost==1.2.0
lightgbm==3.3.5

# Deep Learning (Phases 4-5)
tensorflow==2.12.0
pytorch==2.0.1
transformers==4.28.1

# Statistical & Visualization
numpy==1.24.3
pandas==2.0.2
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
shap==0.41.0

# Utilities
jupyter==1.0.0
tqdm==4.65.0
scikit-optimize==0.9.0  # Bayesian optimization
```

#### Step-by-Step Replication

```bash
# 1. Clone the repository
git clone https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN.git
cd IEEE_EMBS_BHI_25_CSOSEN

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify data is present
ls Track1_Data/  # Should show train.csv, test.csv

# 5. Run full pipeline (WARNING: ~36 hours on CPU)
jupyter notebook Notebooks/

# Run notebooks in order:
# 01_EDA.ipynb → 02_Feature_Engineering.ipynb → 03-07 (Phase experiments)
# → 08_Statistical_Validation.ipynb → 09_Disease_Stratified.ipynb → 10_Interpretability_SHAP.ipynb

# OR use the consolidated analysis notebook:
jupyter notebook sample.ipynb
```

#### Expected Outputs

After running the pipeline, you should see:
- `Results_12W/Conference_Submission/all_results_compiled.json`
- `Results_24W/Conference_Submission/all_results_compiled.json`
- `Reports/figures/` populated with 30+ PNG files
- `All_Trained_Models/` with serialized model artifacts

</details>

---

## 💻 Installation

### Quick Start (Recommended)

```bash
# Clone repository
git clone https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN.git
cd IEEE_EMBS_BHI_25_CSOSEN

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter to explore results
jupyter notebook
```

### Conda Environment (Alternative)

```bash
# Create environment
conda create -n bhi2025 python=3.8
conda activate bhi2025

# Install packages
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Start Guide

#### Option 1: Explore Pre-Computed Results (⭐ Recommended)

All experimental results are already computed and documented. **No need to re-run the entire pipeline** unless you want to modify the experiments.

```bash
# 1. Clone the repository
git clone https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN.git
cd IEEE_EMBS_BHI_25_CSOSEN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter to explore results
jupyter notebook

# 4. Open and run Testing.ipynb to see model predictions
# This notebook loads pre-trained models and makes predictions on test data
```

**What you can explore:**
- ✅ **Pre-trained Models:** All 40 models with optimized hyperparameters in `All_Trained_Models/`
- ✅ **Complete Results:** JSON files with 5-fold CV metrics in `Results_12W/` and `Results_24W/`
- ✅ **Publication Figures:** All visualizations used in the paper in `imgs/`
- ✅ **Predictions:** Run `Testing.ipynb` to generate predictions on test data
- ✅ **Model Registry:** `All_Trained_Models/all_models_summary.json` contains metadata for all models

```bash
# View compiled results (requires jq for pretty JSON)
cat Results_12W/Conference_Submission/all_results_compiled.json | jq .

# Or just view the model summary
cat All_Trained_Models/all_models_summary.json
```

---

#### Option 2: Run Inference with Pre-Trained Models

```bash
# Open the Testing notebook
jupyter notebook Testing.ipynb
```

**The Testing.ipynb notebook provides:**
1. Load pre-trained models from `All_Trained_Models/`
2. Load test data from `Track1_Data/test.csv`
3. Generate predictions for 12-week and 24-week BDI-II scores
4. Export predictions to CSV
5. Visualize model performance

**Example code snippet:**
```python
import pickle
import pandas as pd

# Load best 12-week model (Transformer)
with open('All_Trained_Models/Rank_01_phase5_transformer/model.pkl', 'rb') as f:
    model_12w = pickle.load(f)

# Load test data
test_data = pd.read_csv('Track1_Data/test.csv')

# Make predictions
predictions_12w = model_12w.predict(test_data)
```

---

#### Option 3: Interactive Analysis (Explore Code and Methods)

```bash
# Launch Jupyter
jupyter notebook sample.ipynb
```

**The sample.ipynb notebook includes:**
- ✅ **Data Loading & EDA:** Exploratory analysis of patient characteristics
- ✅ **Feature Engineering:** 26-feature pipeline construction
- ✅ **Model Training Examples:** Code for each of the 5 phases
- ✅ **Cross-Validation:** 5-fold stratified CV implementation
- ✅ **SHAP Analysis:** Feature importance interpretation
- ✅ **Disease Stratification:** Condition-specific model training

**You can modify hyperparameters and re-train individual models without running the full pipeline.**

---

#### Option 4: Full Pipeline Replication (⚠️ Compute-Intensive)

**Warning:** Running all 40 models with hyperparameter tuning takes **~36 hours on CPU** (Intel i7-12650H, 16GB RAM). GPU recommended for Phases 4-5.

```bash
# 1. Clone and install
git clone https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN.git
cd IEEE_EMBS_BHI_25_CSOSEN
pip install -r requirements.txt

# 2. Verify data is present
ls Track1_Data/
# Should show: train.csv, test.csv, data_dictionary.pdf

# 3. Run notebooks sequentially in Notebooks/ folder
jupyter notebook Notebooks/

# Execute in this order:
# 01_EDA.ipynb → 02_Feature_Engineering.ipynb 
# → 03_Phase1_Linear_Models.ipynb → 04_Phase2_Classical_ML.ipynb
# → 05_Phase3_Ensembles.ipynb → 06_Phase4_Deep_Learning.ipynb
# → 07_Phase5_Time_Series.ipynb → 08_Statistical_Validation.ipynb
# → 09_Disease_Stratified.ipynb → 10_Interpretability_SHAP.ipynb
```

**Expected Outputs After Full Run:**
```
Results_12W/Conference_Submission/
├── all_results_compiled.json       # All model metrics (5-fold CV)
├── phase_performance.csv           # Phase-level aggregates
└── best_models_summary.json        # Top 3 performers

Results_24W/Conference_Submission/
├── all_results_compiled.json
├── phase_performance.csv
└── best_models_summary.json

All_Trained_Models/
├── all_models_summary.json         # Model registry
├── Rank_01_phase5_transformer/     # Best 12W model
│   ├── model.pkl
│   ├── hyperparameters.json
│   └── cv_results.json
├── Rank_01_phase3_catboost/        # Best 24W model
│   └── ...
└── [38 more model directories]

imgs/
├── phase_radar_12w_actual.png
├── phase_radar_24w_actual.png
└── [6+ more figures]
```

---

#### Option 5: Run Specific Phases Only

If you want to test a particular modeling approach without running everything:

**Phase 1 (Linear Models) - Fastest, ~30 min:**
```bash
jupyter notebook Notebooks/03_Phase1_Linear_Models.ipynb
```

**Phase 3 (Ensembles) - Best 24W performance, ~2 hours:**
```bash
jupyter notebook Notebooks/05_Phase3_Ensembles.ipynb
```

**Phase 5 (Time-Series) - Best 12W performance, ~8 hours with GPU:**
```bash
jupyter notebook Notebooks/07_Phase5_Time_Series.ipynb
```

**Python Script Example (Phase 3 - CatBoost):**
```python
from SRC_Track1.models.phase3_ensembles import train_catboost
from SRC_Track1.evaluation import cross_validate
from SRC_Track1.data_loader import load_data
from SRC_Track1.preprocessing import engineer_features

# Load and prepare data
train_df = load_data('Track1_Data/train.csv')
X_train, y_train = engineer_features(train_df, target='BDI_24W')

# Train CatBoost with default hyperparameters
model = train_catboost(X_train, y_train)

# Evaluate with 5-fold CV
results = cross_validate(model, X_train, y_train, cv=5)
print(f"CatBoost 24W Performance:")
print(f"  R² = {results['test_r2_mean']:.3f} ± {results['test_r2_std']:.3f}")
print(f"  MAE = {results['test_mae_mean']:.3f} ± {results['test_mae_std']:.3f}")
```

---

### Hardware Requirements by Phase

| Phase | Min RAM | Recommended | GPU Required? | Est. Time (CPU) | Est. Time (GPU) |
|-------|---------|-------------|---------------|-----------------|-----------------|
| **Phase 1** (Linear) | 8GB | 16GB | ❌ No | 30 min | N/A |
| **Phase 2** (Classical ML) | 16GB | 16GB | ❌ No | 2 hours | N/A |
| **Phase 3** (Ensembles) | 16GB | 32GB | ❌ No | 2 hours | N/A |
| **Phase 4** (Deep Learning) | 16GB | 32GB | ⚠️ Recommended | 8 hours | 2 hours |
| **Phase 5** (Time-Series) | 16GB | 32GB | ⚠️ Recommended | 16 hours | 4 hours |
| **Full Pipeline** | 16GB | 32GB | ⚠️ Recommended | **36 hours** | **10 hours** |

**GPU Specs for Deep Learning (Phases 4-5):**
- NVIDIA RTX 3060 (12GB VRAM) or better
- CUDA 11.8+ and cuDNN 8.6+
- TensorFlow 2.12+ with GPU support

**Our Hardware:**
- Acer Nitro 5 Laptop
- Intel i7-12650H (16 CPUs @ ~2.7GHz)
- 16GB RAM
- **No GPU used** (all results generated on CPU)

---

### Troubleshooting

**Issue 1: Out of Memory during Phase 4/5**
```bash
# Reduce batch size in deep learning models
# Edit Notebooks/06_Phase4_Deep_Learning.ipynb
# Change: BATCH_SIZE = 32 → BATCH_SIZE = 8
```

**Issue 2: Missing Dependencies**
```bash
# Reinstall with all extras
pip install -r requirements.txt --upgrade
```

**Issue 3: Bayesian Optimization Timeout**
```bash
# Reduce optimization iterations
# Edit SRC_Track1/models/hyperparameter_tuning.py
# Change: N_ITERATIONS = 100 → N_ITERATIONS = 20
```

**Issue 4: GPU Not Detected (TensorFlow/PyTorch)**
```bash
# Verify GPU setup
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"

# If not detected, install GPU versions:
pip install tensorflow[and-cuda]  # TensorFlow with GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # PyTorch with CUDA 11.8
```

---

### File Structure Guide

```
Key Files to Know:
├── Testing.ipynb                    ← Load pre-trained models and predict
├── sample.ipynb                     ← Interactive analysis (all phases)
├── Track1_Data/train.csv            ← Training data (210 patients)
├── Track1_Data/test.csv             ← Test data for predictions
├── All_Trained_Models/              ← 40 pre-trained models
│   └── all_models_summary.json      ← Model metadata
├── Results_12W/Conference_Submission/
│   └── all_results_compiled.json    ← 12-week CV results
├── Results_24W/Conference_Submission/
│   └── all_results_compiled.json    ← 24-week CV results
└── imgs/                            ← Publication figures
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{sulake2025disease,
  title={Disease-Stratified Depression Risk Prediction Using Multi-Phase Machine Learning},
  author={Sulake, Nikhileswara Rao and Machara, Sai Manikanta Eswar and Katam, Divya},
  booktitle={IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)},
  year={2025},
  organization={IEEE},
  note={Track-1 Data Competition}
}
```

**Paper:** [final-submission.tex](Reports/final-submission.tex) | [PDF](Reports/final-submission.pdf)

---

## 🙏 Acknowledgments

- **IEEE EMBS BHI 2025 Organizers** for providing the dataset and competition framework
- **Patients** who contributed their data to advance mental health research
- **Clinical teams** at participating hospital centers for enabling data collection
- **Open-source community** for ML libraries (scikit-learn, XGBoost, CatBoost, TensorFlow, PyTorch, SHAP)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Team CSOSEN**

- **Lead:** Nikhileswara Rao Sulake (📧 nikhil01446@gmail.com)
- **Institution:** Rajiv Gandhi University of Knowledge Technologies (RGUKT), Nuzvid, India
- **GitHub:** [Nikhil-Rao20](https://github.com/Nikhil-Rao20)

For questions about the methodology, results, or code, please open an [issue](https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN/issues) or contact the team leader directly.

---

<div align="center">

**🏆 IEEE EMBS BHI 2025 · Track-1 · Team CSOSEN**

*Advancing Precision Psychiatry Through Disease-Stratified Machine Learning*

[![GitHub stars](https://img.shields.io/github/stars/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN?style=social)](https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN?style=social)](https://github.com/Nikhil-Rao20/IEEE_EMBS_BHI_25_CSOSEN/network/members)

</div>
