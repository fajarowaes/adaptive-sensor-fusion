# Adaptive Sensor Fusion for Human Activity Recognition (HAR)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MotionSense-green.svg)](https://github.com/mmalekzadeh/motion-sense)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. Project Overview

**Adaptive Sensor Fusion** is an applied artificial intelligence project focused on **Human Activity Recognition (HAR)** using smartphone motion sensor measurements. The project uses the publicly available **MotionSense dataset** to develop and evaluate deep learning architectures capable of recognizing human activities reliably, even when sensors become noisy, degraded, or unavailable.

### Dataset Specifications
- **Source:** [MotionSense Device Motion Data](https://github.com/mmalekzadeh/motion-sense)
- **Device & Placement:** iPhone 6s kept in the participant's front trouser pocket.
- **Sampling Frequency:** 50 Hz ($\Delta t = 20\text{ ms}$).
- **Participants:** 24 diverse participants across varied genders, ages, weights, and heights.
- **Volume:** ~1.4 million sensor observations across 364 raw CSV trials.
- **Target Activities (6 Classes):**
  1. **Downstairs** (`dws` / Class `0`)
  2. **Jogging** (`jog` / Class `1`)
  3. **Sitting** (`sit` / Class `2`)
  4. **Standing** (`std` / Class `3`)
  5. **Upstairs** (`ups` / Class `4`)
  6. **Walking** (`wlk` / Class `5`)

### The 12 Sensor Features
The MotionSense dataset records 12 continuous time-series motion measurements:
1. `attitude.roll`: Rotation angle around the longitudinal axis (radians).
2. `attitude.pitch`: Rotation angle around the lateral axis (radians).
3. `attitude.yaw`: Rotation angle around the vertical axis (radians).
4. `gravity.x`: Directional component of the Earth's gravitational acceleration along the x-axis ($g$).
5. `gravity.y`: Directional component of the Earth's gravitational acceleration along the y-axis ($g$).
6. `gravity.z`: Directional component of the Earth's gravitational acceleration along the z-axis ($g$).
7. `rotationRate.x`: Angular velocity around the x-axis (radians/s).
8. `rotationRate.y`: Angular velocity around the y-axis (radians/s).
9. `rotationRate.z`: Angular velocity around the z-axis (radians/s).
10. `userAcceleration.x`: Dynamic acceleration caused by user motion along the x-axis ($g$).
11. `userAcceleration.y`: Dynamic acceleration caused by user motion along the y-axis ($g$).
12. `userAcceleration.z`: Dynamic acceleration caused by user motion along the z-axis ($g$).

---

## 2. Complete Feature Breakdown: By Project Phase & Member

### A. Phase 2 Features (Data Preprocessing & Exploratory Analysis)

The Phase 2 pipeline prepares, cleans, standardizes, and inspects the raw MotionSense records. All scripts reside in `data scripts/`:

1. **Automated Data Acquisition (`data scripts/download_data.py`):**
   - Automatically downloads `A_DeviceMotion_data.zip` and `data_subjects_info.csv` directly from the official repository via streaming HTTP requests.
   - Saves raw files to `motionsense_output/motionsense_raw.zip`.
   - Exports `motionsense_output/collection_report.json` logging download status and byte sizes.

2. **Data Structuring & Cleaning (`data scripts/clean_data.py`):**
   - In-memory traversal of 364 CSV files across all 24 subjects directly from the compressed archive.
   - Extracts `subject_id` and trial counts from directory paths (`sub_{id}.csv`, `{activity}_{trial}`).
   - Filters out macOS metadata files (`._*`, `__MACOSX`) that corrupt data tables.
   - Numeric conversion and replacement of infinite values (`inf`, `-inf`) with `NaN`.
   - Missing value imputation using feature-wise median statistics.
   - Deduplication based on source file and sample indices.
   - Joins subject demographics (height, weight, age, gender) via validated many-to-one merge.
   - Exports `motionsense_output/motionsense_initial_clean.csv` and `cleaning_summary.json`.

3. **Outlier Management (`data scripts/outlier_analysis.py`):**
   - Implements Interquartile Range ($\text{IQR} = Q_3 - Q_1$) outlier detection.
   - Applies Winsorization (boundary capping) to measurements outside $[Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$ to eliminate extreme sensor artifacts without breaking time continuity.
   - Exports `motionsense_output/motionsense_cleaned.csv` and `outlier_summary.json`.

4. **Feature Preprocessing & Scaling (`data scripts/feature_preprocessing.py`):**
   - Encodes categorical activities into numeric integers ($0$ to $5$) using scikit-learn's `LabelEncoder`.
   - Exports `motionsense_output/activity_label_mapping.json` for deterministic class decoding.
   - Fits global `StandardScaler` across all 12 sensor features: $\tilde{x} = (x - \mu) / \sigma$.
   - Exports `motionsense_output/motionsense_preprocessed.csv`.

5. **Exploratory Data Analysis (`data scripts/eda.py`):**
   - Generates activity distribution bar charts and sensor feature distribution histograms.
   - Computes and renders a $12 \times 12$ Pearson correlation heatmap illustrating gravitational and acceleration relationships.

6. **Phase 2 Pipeline Orchestrator (`data scripts/run_pipeline.py`):**
   - Executes the complete preprocessing workflow in sequence with halt-on-failure safety checks.

---

### B. Phase 3 Features (Member 1: Model Architecture & Implementation)

Member 1 designed, built, and verified the core deep learning architectures in PyTorch:

1. **Framework Selection & Justification:**
   - **Framework:** **PyTorch** (`torch >= 2.0.0`).
   - *Define-by-Run Dynamic Execution:* Enables conditional branch routing, dynamic sensor availability masking ($m_i \in \{0, 1\}$), and reliability gating without static graph recompilation.
   - *Modular Multi-Branch Construction:* Clean object-oriented sub-network encapsulation (`SensorBranchEncoder`, `AdaptiveFusionModule`, `BaselineCNN`, `AdaptiveSensorFusion`).
   - *Optimized 1D Convolutions & Adaptive Pooling:* Temporal feature extraction via `nn.Conv1d` and `nn.AdaptiveAvgPool1d(1)` accommodating arbitrary window lengths ($L=64, 100, 128, 200, 256$).
   - *Decoupled Design:* Architecture implementation is strictly isolated from Member 2's training loop and Member 3's evaluation suite.

2. **Sensor Modality Partitioning (12 Channels $\to$ 2 Physical Domains):**
   - **Branch 1 (Accelerometer & Gravity - 6 Channels):**
     `gravity.x`, `gravity.y`, `gravity.z`, `userAcceleration.x`, `userAcceleration.y`, `userAcceleration.z` (Indices: `[3, 4, 5, 9, 10, 11]`).
   - **Branch 2 (Gyroscope & Attitude - 6 Channels):**
     `attitude.roll`, `attitude.pitch`, `attitude.yaw`, `rotationRate.x`, `rotationRate.y`, `rotationRate.z` (Indices: `[0, 1, 2, 6, 7, 8]`).

3. **Baseline 1D CNN Architecture (`src/model.py`):**
   - Single-stream 1D CNN taking all 12 channels simultaneously.
   - Three 1D conv blocks with Batch Normalization, ReLU, Max-Pooling, Spatial Dropout, Adaptive Average Pooling, and a 2-layer MLP classifier head.
   - Input shape flexibility: accepts `(batch, 12, length)` or auto-permutes `(batch, length, 12)`.
   - **Total Trainable Parameters:** **177,670**.
   - Built-in self-test routine (`test_baseline_cnn()`).

4. **Adaptive Sensor Fusion Architecture (`src/sensor_fusion.py`):**
   - **Total Trainable Parameters:** **265,032**
     - Accelerometer Branch Encoder: 109,440
     - Gyroscope Branch Encoder: 109,440
     - Adaptive Fusion Module: 12,482
     - Classification Head: 33,670
   - **Branch Feature Encoders (`SensorBranchEncoder`):** Independent 3-block 1D CNN encoders outputting 128-d latent representations: $\mathbf{h}_{\text{acc}} \in \mathbb{R}^{B \times 128}$, $\mathbf{h}_{\text{gyro}} \in \mathbb{R}^{B \times 128}$.
   - **Adaptive Fusion Module (`AdaptiveFusionModule`):**
     - Availability masks ($m_{\text{acc}}, m_{\text{gyro}} \in \{0, 1\}$).
     - Continuous reliability scores ($r_{\text{acc}}, r_{\text{gyro}} \in [0, 1]$), with internal learned self-confidence fallback when external reliability is omitted.
     - Effective reliability: $R_i = m_i \cdot r_i$.
     - Content-based importance scoring: $s_i = G(\mathbf{h}_i)$.
     - Normalized dynamic weight formula with division-by-zero protection ($\epsilon = 10^{-8}$):
       $$w_i = \frac{R_i \cdot \exp(s_i - s_{\max})}{\sum_k R_k \cdot \exp(s_k - s_{\max}) + \epsilon}$$
     - **Boundary Guarantees:**
       - Accelerometer failure ($m_{\text{acc}}=0$) $\implies w_{\text{acc}}=0.0, \; w_{\text{gyro}}=1.0$ (100% reliance on Gyroscope).
       - Gyroscope failure ($m_{\text{gyro}}=0$) $\implies w_{\text{gyro}}=0.0, \; w_{\text{acc}}=1.0$ (100% reliance on Accelerometer).
       - Total sensor failure ($m_{\text{acc}}=0, m_{\text{gyro}}=0$) $\implies w_{\text{acc}}=0.5, \; w_{\text{gyro}}=0.5$ (neutral fallback).
     - **Corrupted Signal Elimination:** Multiplies latent representations by availability masks: $\mathbf{h}_i^{\text{eff}} = \mathbf{h}_i \cdot m_i$ to ensure noise or invalid readings are zeroed out before classification.
     - Fused representation: $\mathbf{h}_{\text{fused}} = [w_{\text{acc}} \cdot \mathbf{h}_{\text{acc}}^{\text{eff}} \;\|\; w_{\text{gyro}} \cdot \mathbf{h}_{\text{gyro}}^{\text{eff}}] \in \mathbb{R}^{B \times 256}$.
   - Dual input support: unified 12-channel tensor or separate 6-channel branch streams.
   - Built-in 7-scenario verification test suite (`test_adaptive_sensor_fusion()`).

5. **Architecture Diagram & Verification:**
   - `test_models.py`: Root demonstration script verifying forward passes, predictions, dynamic weight shifts, and sensor dropouts.
   - `diagrams/model_architecture.png`: 300 DPI comparative diagram illustrating both architectures.
   - `Reports/model_architecture.tex`: Academic LaTeX report section detailing model design and mathematical formulation.

---

### C. Phase 3 & 4 Features (Member 2: Data Windowing, Training, Tuning, Degradation & Reporting)

Member 2 designed, implemented, and executed the entire data partitioning, training, hyperparameter optimization, sensor degradation simulation, and experimental analysis:

1. **Leak-Free Data Pipeline & Boundary-Guarded Windowing (`src/data_loader.py`):**
   - **Subject-Wise 70/15/15 Disjoint Partitioning:** 24 participants partitioned with fixed seed (42):
     - **Train (17 subjects, ~70.83%):** `[1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 16, 17, 18, 19, 22, 23]` (19,562 windows)
     - **Val (3 subjects, 12.50%):** `[8, 21, 24]` (3,545 windows)
     - **Test (4 subjects, ~16.67%):** `[7, 11, 15, 20]` (4,613 windows)
     - Disjoint assertions verified; split persisted to `results/subject_split.json`.
   - **Leak-Free Scaling:** Fitted a new `StandardScaler` strictly on raw 12-channel readings of the 17 training subjects, saving `models/train_only_scaler.pkl`. Val and Test are transformed strictly using training statistics to prevent data leakage.
   - **Boundary-Guarded Windowing:** Slices time-series into windows of $W=100$ samples (2.0 s at 50 Hz) with stride $S=50$ (50% overlap). Window slicing operates strictly within `(participant, activity, trial, source_file)` groups so temporal windows never cross trials or subjects.
   - **Dataset & Caching:** PyTorch `MotionSenseDataset` returning $(x, y)$ with $x \in \mathbb{R}^{12 \times 100}$ (float32) and $y \in \{0, \dots, 5\}$ (int64). Caches 27,720 processed windows to `motionsense_output/windows_w100_s50_seed42.npz`. Full summary logged in `results/data_summary.json`.

2. **Sensor Degradation Simulation Engine (`src/sensor_degradation.py`):**
   - **Strict Non-Destructive Invariant:** Tensors are always cloned before corruptions; never mutates memory in-place.
   - **6 Physical Degradation Conditions:**
     - `normal`: Clean baseline signal ($\sigma=0.0$).
     - `noisy_acc`: Gaussian noise on accelerometer channels (default $\sigma=0.5$).
     - `noisy_gyro`: Gaussian noise on gyroscope channels (default $\sigma=0.5$).
     - `missing_acc`: Accelerometer failure (channels zeroed to standardized mean; $m_{\text{acc}}=0.0$).
     - `missing_gyro`: Gyroscope failure (channels zeroed to standardized mean; $m_{\text{gyro}}=0.0$).
     - `noisy_both`: Elevated Gaussian noise on all 12 channels ($\sigma=1.0$).
   - **Deterministic Generator Seeding:** Per-(condition, seed) `torch.Generator` guarantees bit-identical noise across all models.
   - **Universal Forward Wrapper (`forward_with_condition`):** Automatically detects architecture type and delivers availability masks and SNR reliability indicators.
   - **Online Robust Augmentation (`degrade_batch_random`):** Dynamic per-sample stochastic corruption for robust training.

3. **Training & Validation Engine (`src/train.py`):**
   - **Model Factory (`build_model`):** Modular factory supporting uniform post-initialization dropout overrides.
   - **Post-Init Dropout Override (`set_uniform_dropout`):** Traverses model modules to override dropout rates post-initialization, preserving Member 1's code while enabling rigorous tuning.
   - **Training Loop:** CrossEntropyLoss, Adam / SGD with momentum, early stopping with patience $P=7$ epochs and automatic best validation weight restoration.
   - **Dual Training Modes:** Clean training (`trained_on_normal`) and robust augmentation (`degradation_aware`).
   - **Convergence Curves (`plot_curves`):** Renders 300 DPI loss and accuracy curves (`results/loss_curve.png`, `results/accuracy_curve.png`) parsed directly from `results/training_history.csv`.

4. **Systematic Hyperparameter Optimization (`src/hyperparameter_tuning.py`):**
   - **Search Space (162 Configs per Model):** $\text{lr} \in \{0.001, 0.0005, 0.0001\}$, $\text{batch\_size} \in \{32, 64, 128\}$, $\text{dropout} \in \{0.2, 0.3, 0.5\}$, $\text{max\_epochs} \in \{20, 30, 50\}$, $\text{optimizer} \in \{\text{adam}, \text{sgd}\}$.
   - **Stratified Random Search (20 Runs per Model):** Latin-hypercube inspired coverage guaranteeing all 14 discrete dimension values appear in the sample with seed=42.
   - **Crash-Safe & Resumable:** Writes CSV rows immediately in append mode; set-based duplicate detection skips finished runs.
   - **Selection Strictly on Validation Set:** Selected optimal configs without ever accessing the test set (`results/best_hyperparameters.json`):
     - **BaselineCNN:** $\text{lr}=0.0005$, $\text{batch\_size}=64$, $\text{dropout}=0.3$, $\text{optimizer}=\text{adam}$, $\text{max\_epochs}=30$.
     - **AdaptiveSensorFusion:** $\text{lr}=0.001$, $\text{batch\_size}=32$, $\text{dropout}=0.5$, $\text{optimizer}=\text{adam}$, $\text{max\_epochs}=30$.
   - **Tuning Visualization:** Validation accuracy heatmap saved to `results/hyperparameter_summary.png`. Per-run metrics cleanly isolated in `results/tuning_metrics/`.

5. **Final Experiments & Robustness Benchmarking (`src/run_experiments.py`):**
   - **Full Pipeline Orchestrator:** Modular CLI supporting training, evaluation, sigma sweep, plotting, and factual summary.
   - **Checkpoint Persistence:**
     - `models/baseline_best.pth` & `models/baseline_config.json`
     - `models/adaptive_best.pth` & `models/adaptive_config.json`
     - `models/baseline_degradation_aware.pth` & `models/baseline_degradation_aware_config.json`
     - `models/adaptive_degradation_aware.pth` & `models/adaptive_degradation_aware_config.json`
   - **Degradation Benchmarking:** Evaluates all 4 models across all 6 conditions on Validation and Test sets ($N=4,613$ test windows), logging Accuracy, Precision, Recall, and F1 to `results/degradation_results.csv`.
   - **Dynamic Weight Allocation Analysis:** Records $w_{\text{acc}}$ and $w_{\text{gyro}}$ shifts to `results/adaptive_fusion_weights.csv`.
   - **Noise Sensitivity Sweep:** Sweeps $\sigma \in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]$ across noisy conditions on test data (`results/noise_sweep.csv`).
   - **Publication-Quality Figures (300 DPI, all saved in `results/`):**
     - `degradation_accuracy_comparison.png`: Grouped bars of Accuracy across conditions for clean vs. augmented training.
     - `degradation_f1_comparison.png`: Grouped bars of Macro F1 across conditions.
     - `noise_sweep.png`: 3-panel line trajectory showing accuracy drop across increasing noise levels ($\sigma = 0 \to 2.0$).
     - `confusion_matrix_normal_baseline.png` & `confusion_matrix_normal_adaptive.png`: Row-normalized confusion matrices on clean test data.
     - `adaptive_fusion_weights.png`: Bar chart demonstrating dynamic weight reallocation (100% shift to active sensor under failure).

6. **Integration Documentation & Checkpoint Verification (`docs/MEMBER2_INTEGRATION.md`, `src/verify_checkpoints.py`):**
   - Detailed integration documentation providing folder structure, input contracts, constructor arguments, DataLoader usage for Member 3, and degradation snippets.
   - Automated verification script confirming all saved checkpoints load, accept `(B, 12, 100)` float32 inputs, return `(B, 6)` logits, produce valid predictions (0–5), and dynamically shift weights ($w_{\text{acc}}=0.0, w_{\text{gyro}}=1.0$ under `missing_acc`).

7. **Automated LaTeX Reporting (`src/generate_latex_tables.py`, `Reports/training_and_tuning.tex`):**
   - Python script generating clean booktabs `.tex` table fragments directly from CSV/JSON outputs (`Reports/tables/split_table.tex`, `hparam_table.tex`, `degradation_table.tex`, `weights_table.tex`).
   - Academic LaTeX report section covering data splitting, window generation, training configuration, hyperparameter optimization, convergence curves, degradation benchmarking, dynamic weights analysis, and study limitations. Narrative text references auto-generated tables and figures without hardcoded numbers.

---

## 3. Project Directory Structure

```text
adaptive-sensor-fusion/
├── README.md                                  # Complete project documentation
├── requirements.txt                           # Project dependencies (PyTorch, Matplotlib, etc.)
├── test_models.py                             # Standalone model verification demo
├── adaptive_sensor_fusion.ipynb                # Interactive research notebook
├── docs/                                      # Integration Documentation
│   ├── MEMBER1_INTEGRATION.md                 # Member 1 architecture integration guide
│   └── MEMBER2_INTEGRATION.md                 # Member 2 data, training & evaluation guide
├── data scripts/                              # Phase 2 Preprocessing & EDA Pipeline
│   ├── download_data.py                       # Download raw MotionSense ZIP & subjects info
│   ├── clean_data.py                          # Data structuring, cleaning & imputation
│   ├── outlier_analysis.py                    # IQR detection & outlier capping
│   ├── feature_preprocessing.py               # Label encoding & StandardScaler fitting
│   ├── eda.py                                 # Activity distributions & correlation heatmap
│   └── run_pipeline.py                        # Sequential Phase 2 pipeline orchestrator
├── src/                                       # Core Source Code Package
│   ├── __init__.py                            # Module exports & channel constants
│   ├── model.py                               # Baseline 1D CNN architecture (Member 1)
│   ├── sensor_fusion.py                       # Adaptive Sensor Fusion architecture (Member 1)
│   ├── generate_diagram.py                    # Architecture diagram generator (Member 1)
│   ├── data_loader.py                         # Windowing, scaling & splitting (Member 2)
│   ├── sensor_degradation.py                  # Non-destructive degradation simulation (Member 2)
│   ├── train.py                               # Training engine, dropout override, curves (Member 2)
│   ├── hyperparameter_tuning.py               # Stratified random search tuner (Member 2)
│   ├── run_experiments.py                     # Pipeline orchestrator: training, eval, plots (Member 2)
│   ├── verify_checkpoints.py                  # Checkpoint and inference verification (Member 2)
│   └── generate_latex_tables.py               # Auto-generates LaTeX tables from CSVs (Member 2)
├── models/                                    # Serialized Checkpoints & Metadata
│   ├── baseline_best.pth                      # Best BaselineCNN weights (clean training)
│   ├── baseline_config.json                   # Hyperparameter metadata for baseline_best
│   ├── adaptive_best.pth                      # Best AdaptiveSensorFusion weights (clean training)
│   ├── adaptive_config.json                   # Hyperparameter metadata for adaptive_best
│   ├── baseline_degradation_aware.pth         # BaselineCNN weights (robust augmentation)
│   ├── baseline_degradation_aware_config.json # Metadata for baseline_degradation_aware
│   ├── adaptive_degradation_aware.pth         # AdaptiveSensorFusion weights (robust augmentation)
│   ├── adaptive_degradation_aware_config.json # Metadata for adaptive_degradation_aware
│   └── train_only_scaler.pkl                  # Leak-free StandardScaler on training subjects only
├── results/                                   # Experimental Results, Curves & Visualizations
│   ├── subject_split.json                     # Disjoint 70/15/15 participant partition
│   ├── data_summary.json                      # Window counts & class distribution per split
│   ├── best_hyperparameters.json              # Selected optimal hyperparameters per model
│   ├── hyperparameter_results.csv             # Full hyperparameter sweep logs (40 evaluated runs)
│   ├── hyperparameter_summary.png             # Tuning validation accuracy heatmap
│   ├── training_history.csv                   # Per-epoch training and validation loss/acc history
│   ├── loss_curve.png                         # 300 DPI CrossEntropy loss curve
│   ├── accuracy_curve.png                     # 300 DPI classification accuracy curve
│   ├── degradation_results.csv                # Accuracy, Precision, Recall, F1 across 6 conditions
│   ├── noise_sweep.csv                        # Performance trajectory across sigma in [0.0, 2.0]
│   ├── adaptive_fusion_weights.csv            # Dynamic fusion branch weights (w_acc, w_gyro)
│   ├── degradation_accuracy_comparison.png    # 300 DPI grouped bar chart: Accuracy across conditions
│   ├── degradation_f1_comparison.png          # 300 DPI grouped bar chart: Macro F1 across conditions
│   ├── noise_sweep.png                        # 300 DPI 3-panel noise sensitivity line trajectories
│   ├── confusion_matrix_normal_baseline.png   # 300 DPI normalized test confusion matrix (Baseline)
│   ├── confusion_matrix_normal_adaptive.png   # 300 DPI normalized test confusion matrix (Adaptive)
│   ├── adaptive_fusion_weights.png            # 300 DPI dynamic weight reallocation bar chart
│   └── tuning_metrics/                        # Isolated individual JSON logs from tuning runs
├── diagrams/                                  # Visual Architecture Artifacts
│   └── model_architecture.png                 # 300 DPI high-resolution architecture diagram
├── Reports/                                   # Project Reports & Deliverables
│   ├── Adaptive_sensor_Report .pdf            # Phase 2 existing report
│   ├── model_architecture.tex                 # Member 1 LaTeX section fragment
│   ├── training_and_tuning.tex                # Member 2 LaTeX section fragment
│   └── tables/                                # Auto-generated booktabs table fragments
│       ├── split_table.tex                    # Dataset partition & window counts
│       ├── hparam_table.tex                   # Top hyperparameter configurations
│       ├── degradation_table.tex              # Test set degradation performance benchmark
│       └── weights_table.tex                  # Dynamic fusion branch weights summary
└── motionsense_output/                        # Generated locally (ignored by Git)
    ├── motionsense_preprocessed.csv           # Clean preprocessed sensor dataset (628 MB)
    ├── windows_w100_s50_seed42.npz            # Cached 27,720 temporal windows
    └── activity_label_mapping.json            # Target activity integer mapping (0-5)
```

---

## 4. Installation & Requirements

- **Python Version:** Python 3.10+ (tested on Python 3.12)
- **Key Dependencies:**
  - `torch>=2.0.0`
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `joblib>=1.3.0`
  - `requests>=2.31.0`

Install all dependencies from the repository root:

```bash
py -m pip install -r requirements.txt
```

---

## 5. How to Run

### A. Verify Model Checkpoints & Inference Contract
Verifies that all four trained checkpoints load, accept the input shape `(B, 12, 100)`, and produce `(B, 6)` logits:
```powershell
py src/verify_checkpoints.py
```

### B. Execute Full Experiments Pipeline (End-to-End)
Trains models under optimal hyperparameters, evaluates all 6 degradation conditions on Val and Test sets, runs the $\sigma$-sweep, generates all 300-DPI publication plots, and prints an objective summary:
```powershell
py src/run_experiments.py --all
```

### C. Run Specific Pipeline Stages
```powershell
# 1. Train models only using selected hyperparameters
py src/run_experiments.py --train

# 2. Skip training and evaluate existing checkpoints on degradation conditions
py src/run_experiments.py --skip-train --eval-degradation

# 3. Run noise sigma-sweep on test set
py src/run_experiments.py --skip-train --sigma-sweep

# 4. Regenerate only plots and summary from existing CSVs
py src/run_experiments.py --plot --summary

# 5. Multi-seed training (reporting mean +/- std across seeds 42, 43, 44)
py src/run_experiments.py --train --seeds 42,43,44

# 6. Rapid smoke test verification (~15 seconds)
py src/run_experiments.py --smoke
```

### D. Run Hyperparameter Tuning
Performs a stratified random search across the 162-combination space (20 runs per model):
```powershell
py src/hyperparameter_tuning.py --mode random --max-runs 20 --models both
```

### E. Regenerate LaTeX Tables
Automatically regenerates all `.tex` table fragments in `Reports/tables/` directly from the CSV and JSON artifacts:
```powershell
py src/generate_latex_tables.py
```

### F. Run Phase 3 Model Verification Demos
```powershell
# 1. Quick end-to-end demo test
py test_models.py

# 2. Baseline 1D CNN test suite
py src/model.py

# 3. Adaptive Sensor Fusion test suite (7 scenarios)
py src/sensor_fusion.py

# 4. Regenerate architecture diagram
py src/generate_diagram.py
```

### G. Run Phase 2 Data Preprocessing (if reproducing from scratch)
```powershell
py "data scripts/run_pipeline.py"
```

---

## 6. Team Integration & Cross-Member Workflows

Complete integration documentation is available in [`docs/MEMBER2_INTEGRATION.md`](docs/MEMBER2_INTEGRATION.md).

### How to Load Trained Models
```python
import json
import torch
from src.model import load_model
from src.sensor_fusion import load_adaptive_model
from src.train import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load BaselineCNN
with open("models/baseline_config.json") as f:
    cfg_base = json.load(f)
model_base = build_model("baseline", dropout=cfg_base["hyperparameters"]["dropout"]).to(device)
load_model(model_base, "models/baseline_best.pth", map_location=str(device))
model_base.eval()

# 2. Load AdaptiveSensorFusion
with open("models/adaptive_config.json") as f:
    cfg_adapt = json.load(f)
model_adapt = build_model("adaptive", dropout=cfg_adapt["hyperparameters"]["dropout"]).to(device)
load_adaptive_model(model_adapt, "models/adaptive_best.pth", map_location=str(device))
model_adapt.eval()
```

### How Member 3 Loads the Identical Dataset Partition
```python
from src.data_loader import get_dataloaders, load_split

# Load exact participant IDs (Train: 17, Val: 3, Test: 4)
split = load_split("results/subject_split.json")

# Instantiate PyTorch DataLoaders (returns x shaped (12, 100), float32)
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=64,
    seed=42,
    num_workers=0,
)
```

### How to Apply Sensor Degradation
```python
from src.sensor_degradation import forward_with_condition

with torch.no_grad():
    # Evaluate under accelerometer disconnection (mask_acc=0.0)
    preds, weights = forward_with_condition(
        model=model_adapt,
        x=batch_x,
        condition="missing_acc",
        seed=42,
        return_weights=True,
        mode="predict",
    )
    # Model shifts 100% reliance to Gyroscope
    print("w_acc:", weights["w_acc"].mean().item())   # 0.0
    print("w_gyro:", weights["w_gyro"].mean().item()) # 1.0
```

---

## 7. Experimental Results Summary

Evaluated on the unseen test set ($N = 4,613$ temporal windows from 4 holdout participants):

### A. Clean Training (`trained_on_normal`)

| Condition | Baseline Acc (%) | Adaptive Acc (%) | $\Delta$ Acc (%) | Baseline F1 | Adaptive F1 | $\Delta$ F1 | Superior Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pristine (Normal)** | 93.93% | 92.48% | -1.45% | 0.9230 | 0.9096 | -0.0134 | Baseline (-1.45%) |
| **Noisy Accel ($\sigma=0.5$)** | 92.70% | 92.13% | -0.56% | 0.9041 | 0.8927 | -0.0113 | Baseline (-0.56%) |
| **Noisy Gyro ($\sigma=0.5$)** | 93.22% | 92.00% | -1.21% | 0.9112 | 0.9042 | -0.0070 | Baseline (-1.21%) |
| **Missing Accel ($m_{\text{acc}}=0$)** | 43.90% | **63.97%** | **+20.07%** | 0.3333 | **0.4670** | **+0.1337** | **Adaptive (+20.07%)** |
| **Missing Gyro ($m_{\text{gyro}}=0$)** | 65.25% | 52.63% | -12.62% | 0.4804 | 0.4818 | +0.0014 | Baseline (-12.62%) |
| **Noisy Both ($\sigma=1.0$)** | 10.47% | 10.41% | -0.06% | 0.0551 | 0.0570 | +0.0019 | Parity ($\approx$) |

### B. Robust Augmentation (`degradation_aware`)

| Condition | Baseline Acc (%) | Adaptive Acc (%) | $\Delta$ Acc (%) | Baseline F1 | Adaptive F1 | $\Delta$ F1 | Superior Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pristine (Normal)** | 92.91% | 92.30% | -0.61% | 0.9236 | 0.9124 | -0.0111 | Baseline (-0.61%) |
| **Noisy Accel ($\sigma=0.5$)** | 92.85% | 92.07% | -0.78% | 0.9227 | 0.9079 | -0.0147 | Baseline (-0.78%) |
| **Noisy Gyro ($\sigma=0.5$)** | 92.87% | 91.70% | -1.17% | 0.9225 | 0.9072 | -0.0153 | Baseline (-1.17%) |
| **Missing Accel ($m_{\text{acc}}=0$)** | 90.44% | 90.18% | -0.26% | 0.8865 | 0.8869 | +0.0004 | Parity ($\approx$) |
| **Missing Gyro ($m_{\text{gyro}}=0$)** | 93.41% | 91.63% | -1.78% | 0.9106 | 0.8961 | -0.0145 | Baseline (-1.78%) |
| **Noisy Both ($\sigma=1.0$)** | 87.41% | 76.35% | -11.06% | 0.8572 | 0.7352 | -0.1220 | Baseline (-11.06%) |

### C. Dynamic Fusion Weight Behavior ($w_{\text{acc}}$ vs. $w_{\text{gyro}}$)
- **Complete Accelerometer Disconnection (`missing_acc`):** $w_{\text{acc}} = 0.0000$, $w_{\text{gyro}} = 1.0000$ (100% reliance dynamically shifted to Gyroscope branch).
- **Complete Gyroscope Disconnection (`missing_gyro`):** $w_{\text{acc}} = 1.0000$, $w_{\text{gyro}} = 0.0000$ (100% reliance dynamically shifted to Accelerometer branch).
- **Under Clean Data:** $w_{\text{acc}} \approx 0.634, w_{\text{gyro}} \approx 0.366$ (favoring dynamic linear acceleration for cyclical activities).
- **Under Degradation-Aware Training:** $w_{\text{acc}} \approx 0.503, w_{\text{gyro}} \approx 0.497$ (balanced multi-modal contribution).

---

## 8. Team Contributions

- **Member 1 (Architecture & Implementation - Completed):**
  - Designed and implemented `BaselineCNN` (`src/model.py`) and dual-branch `AdaptiveSensorFusion` (`src/sensor_fusion.py`).
  - Implemented dynamic reliability gating, sensor availability masking, and zero-division guards.
  - Built unit tests, the architecture diagram (`diagrams/model_architecture.png`), dependencies (`requirements.txt`), demo script (`test_models.py`), and the academic LaTeX report section (`Reports/model_architecture.tex`).
- **Member 2 (Data Pipeline, Training, Tuning, Degradation & Reporting - Completed):**
  - Designed leak-free subject-wise 70/15/15 splitting (`results/subject_split.json`) and fitted train-only `StandardScaler` (`models/train_only_scaler.pkl`).
  - Implemented boundary-guarded temporal windowing ($W=100, S=50$) generating 27,720 windows (`src/data_loader.py`).
  - Created reproducible, non-destructive sensor degradation simulation engine across 6 physical conditions (`src/sensor_degradation.py`).
  - Built training engine with early stopping, post-init dropout overrides, and publication curves (`src/train.py`).
  - Implemented stratified random hyperparameter search across 162 configurations (`src/hyperparameter_tuning.py`).
  - Built full experiment orchestration pipeline for training, 6-condition benchmarks, and sigma sweeps (`src/run_experiments.py`).
  - Authored integration documentation (`docs/MEMBER2_INTEGRATION.md`) and verified checkpoints (`src/verify_checkpoints.py`).
  - Automated LaTeX table generation (`src/generate_latex_tables.py`) and authored academic LaTeX report section (`Reports/training_and_tuning.tex`).
- **Member 3 (Independent Evaluation & Final Report Integration):**
  - Responsible for independent test verification, cross-validation evaluation, and compiling final report.

---

## 9. Data & Repository Policy

Raw datasets and generated output files (`motionsense_output/`) are excluded from Git version control via `.gitignore`. The repository maintains all code, models, diagrams, reports, and documentation required for complete reproducibility.
