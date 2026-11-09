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
  2. **Upstairs** (`ups` / Class `1`)
  3. **Walking** (`wlk` / Class `2`)
  4. **Jogging** (`jog` / Class `3`)
  5. **Standing** (`std` / Class `4`)
  6. **Sitting** (`sit` / Class `5`)

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

## 2. Complete Feature Breakdown: Old vs. New

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
   - Fits `StandardScaler` ($\mu = 0, \sigma = 1$) across all 12 sensor features:
     $$\tilde{x} = \frac{x - \mu}{\sigma}$$
   - Serializes fitted scaler to `motionsense_output/sensor_scaler.pkl` for transforming future test windows.
   - Exports `motionsense_output/motionsense_preprocessed.csv`.

5. **Exploratory Data Analysis (`data scripts/eda.py`):**
   - Generates activity distribution bar charts.
   - Plots sensor feature distribution histograms.
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
   - *Optimized 1D Convolutions & Adaptive Pooling:* Temporal feature extraction via `nn.Conv1d` and `nn.AdaptiveAvgPool1d(1)` accommodating arbitrary window lengths ($L=64, 128, 200, 256$).
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

5. **Quick-Run Verification Script (`test_models.py`):**
   - Standalone root demonstration script verifying forward pass, predictions, dynamic weight shifts, and simulated sensor dropout.

6. **Publication-Quality Architecture Diagram (`diagrams/model_architecture.png` & `src/generate_diagram.py`):**
   - 300 DPI comparative diagram illustrating both architectures side-by-side.

7. **LaTeX Report Section (`Reports/model_architecture.tex`):**
   - Ready-to-compile academic section detailing framework selection, layer specifications, mathematical formulation of adaptive fusion, loss function, optimizer, and reliability analysis.

8. **Package Initialization & Dependencies (`src/__init__.py`, `requirements.txt`):**
   - Standard Python package export for clean imports.
   - Pinned dependencies for PyTorch, Matplotlib, and core data science libraries.

---

## 3. Project Directory Structure

```text
adaptive-sensor-fusion/
├── README.md                           # Comprehensive project documentation
├── .gitignore                          # Git ignore policy
├── requirements.txt                    # Project dependencies (PyTorch, Matplotlib, etc.)
├── test_models.py                      # Standalone demonstration & verification script
├── adaptive_sensor_fusion.ipynb         # Interactive research notebook
├── data scripts/                       # Phase 2 Preprocessing & EDA Pipeline
│   ├── download_data.py                # Download raw MotionSense ZIP & subjects info
│   ├── clean_data.py                   # Data structuring, cleaning & imputation
│   ├── outlier_analysis.py             # IQR detection & outlier capping
│   ├── feature_preprocessing.py        # Label encoding & StandardScaler fitting
│   ├── eda.py                          # Activity distributions & correlation heatmap
│   └── run_pipeline.py                 # Sequential Phase 2 pipeline orchestrator
├── src/                                # Phase 3 Core Deep Learning Architectures
│   ├── __init__.py                     # Module exports & channel constants
│   ├── model.py                        # Baseline 1D CNN (177,670 parameters)
│   ├── sensor_fusion.py                # Adaptive Sensor Fusion Model (265,032 parameters)
│   └── generate_diagram.py             # Matplotlib architecture diagram generator
├── diagrams/                           # Visual Architecture Artifacts
│   └── model_architecture.png          # 300 DPI high-resolution architecture diagram
├── Reports/                            # Project Reports & Deliverables
│   ├── Adaptive_sensor_Report .pdf     # Phase 2 existing report
│   └── model_architecture.tex          # Phase 3 Member 1 LaTeX report section
└── motionsense_output/                 # Generated locally (ignored by Git)
    ├── motionsense_raw.zip
    ├── data_subjects_info.csv
    ├── motionsense_cleaned.csv
    ├── motionsense_preprocessed.csv
    ├── collection_report.json
    ├── cleaning_summary.json
    ├── outlier_summary.json
    ├── activity_label_mapping.json
    ├── sensor_scaler.pkl
    └── EDA visualizations
```

---

## 4. Installation & Requirements

- **Python Version:** Python 3.10+ (tested up to Python 3.14)
- **Key Dependencies:**
  - `torch>=2.0.0`
  - `matplotlib>=3.7.0`
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `joblib>=1.3.0`
  - `requests>=2.31.0`

Install all dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

---

## 5. How to Run

### A. Run Phase 3 Model Verifications

Run the built-in test suites to verify shapes, parameters, and dynamic reliability weighting:

```bash
# 1. Quick end-to-end demo test
python test_models.py

# 2. Comprehensive Baseline 1D CNN test suite
python src/model.py

# 3. Comprehensive Adaptive Sensor Fusion test suite (7 scenarios)
python src/sensor_fusion.py

# 4. Re-generate the high-resolution architecture diagram
python src/generate_diagram.py
```

### B. Run Phase 2 Data Preprocessing Pipeline (if reproducing from scratch)

```bash
python "data scripts/run_pipeline.py"
```

---

## 6. Team Integration Guide

### For Member 2 (Training Pipeline):
```python
import torch
from src.model import BaselineCNN, save_model
from src.sensor_fusion import AdaptiveSensorFusion, save_adaptive_model

# 1. Instantiate models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
baseline_model = BaselineCNN().to(device)
adaptive_model = AdaptiveSensorFusion().to(device)

# 2. Access loss criterion and optimizer
criterion = adaptive_model.get_loss_criterion()                     # CrossEntropyLoss()
optimizer = adaptive_model.get_optimizer(lr=1e-3, weight_decay=1e-4) # Adam

# 3. Forward pass in training loop:
# Input x: (batch_size, 12, sequence_length), e.g., (64, 12, 128)
# Target y: (batch_size,) integer class indices 0-5
logits = adaptive_model(x)
loss = criterion(logits, y)

# 4. Save trained checkpoints
save_model(baseline_model, "models/baseline_cnn.pth")
save_adaptive_model(adaptive_model, "models/adaptive_fusion.pth")
```

### For Member 3 (Evaluation & Stress-Testing):
```python
from src.sensor_fusion import AdaptiveSensorFusion, load_adaptive_model

model = AdaptiveSensorFusion()
load_adaptive_model(model, "models/adaptive_fusion.pth", map_location="cpu")

# Predict class labels (0 to 5)
preds = model.predict(x)

# Predict probability distributions
probs = model.predict_proba(x)

# Simulate accelerometer failure (e.g., sensor disconnected)
preds_no_acc = model.predict(x, mask_acc=0.0, mask_gyro=1.0)

# Simulate gyroscope failure
preds_no_gyro = model.predict(x, mask_acc=1.0, mask_gyro=0.0)

# Simulate noisy sensor (e.g. low SNR on accelerometer)
preds_noisy = model.predict(x, reliability_acc=0.2, reliability_gyro=0.95)

# Inspect dynamic weights
logits, weights = model(x, return_weights=True)
print("Accel Weight:", weights["w_acc"].mean().item())
print("Gyro Weight:", weights["w_gyro"].mean().item())
```

---

## 7. Team Contributions

- **Member 1 (Architecture & Implementation - Completed):**
  - Designed and implemented `BaselineCNN` (`src/model.py`) and dual-branch `AdaptiveSensorFusion` (`src/sensor_fusion.py`).
  - Implemented dynamic reliability gating, sensor availability masking, and zero-division guards.
  - Built unit tests, the architecture diagram (`diagrams/model_architecture.png`), dependencies (`requirements.txt`), demo script (`test_models.py`), and the academic LaTeX report section (`Reports/model_architecture.tex`).
- **Member 2 (Training & Validation Pipeline):**
  - Responsible for sequence windowing, data loaders, training loops, validation curves, and model training.
- **Member 3 (Evaluation & Robustness Benchmark):**
  - Responsible for testing trained models, computing classification metrics (Accuracy, Precision, Recall, F1-score), confusion matrices, and running noise/availability stress-tests.

---

## 8. Data & Repository Policy

Raw datasets and generated output files (`motionsense_output/`) are excluded from Git version control via `.gitignore`. The repository maintains all code, models, diagrams, reports, and documentation required for complete reproducibility.
