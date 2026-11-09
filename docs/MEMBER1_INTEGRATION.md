# Comprehensive Project Documentation: Adaptive Sensor Fusion for Human Activity Recognition (HAR)

This document provides a detailed breakdown of all **existing (old) features** from Phase 2 (Data Preprocessing & EDA) and all **newly implemented features** from Phase 3 (Member 1: Model Architecture & Implementation).

---

## 1. Project Overview

- **Project Title:** Adaptive Sensor Fusion for Human Activity Recognition (HAR)
- **Dataset:** MotionSense Dataset (iPhone 6s captured in front pocket, 50 Hz sampling rate)
- **Scale:** ~1.4 million sensor observations across 364 raw CSV trials from 24 participants
- **Target Activities (6 classes):**
  1. Downstairs (`dws`)
  2. Upstairs (`ups`)
  3. Walking (`wlk`)
  4. Jogging (`jog`)
  5. Standing (`std`)
  6. Sitting (`sit`)
- **Primary Objective:** Build a deep learning activity recognition system capable of maintaining high classification reliability even when individual sensor groups become noisy, corrupted, or completely unavailable.

---

## 2. Old Features: Phase 2 Data Pipeline & Exploratory Analysis

The following features were established in Phase 2 to collect, clean, standardize, and inspect the raw sensor data:

### A. Data Acquisition (`data scripts/download_data.py`)
- **Automated Data Retrieval:** Downloads the official MotionSense raw archive (`A_DeviceMotion_data.zip`) and participant metadata (`data_subjects_info.csv`) directly from GitHub via HTTP streams.
- **Local Persistence:** Saves the files to `motionsense_output/motionsense_raw.zip` without requiring manual browser downloads.
- **Collection Reporting:** Generates `motionsense_output/collection_report.json` with download timestamps, byte sizes, and HTTP response statuses.

### B. Data Cleaning and Structuring (`data scripts/clean_data.py`)
- **Direct Zip Streaming:** Reads 364 CSV files directly from the ZIP archive in memory without extracting millions of loose files to disk.
- **Trial & Metadata Extraction:**
  - Extracts `subject_id` (1–24) and trial index from directory and filenames (`sub_{id}.csv`, `{activity}_{trial}`).
  - Maps short activity codes (`dws`, `ups`, `wlk`, `jog`, `std`, `sit`) into human-readable activity labels.
- **macOS Metadata Filtering:** Skips macOS hidden files (`._*`, `__MACOSX`) that corrupt data concatenation.
- **Data Hygiene & Missing Value Imputation:**
  - Converts all 12 sensor columns to numeric values (`pd.to_numeric`).
  - Replaces infinite values (`inf`, `-inf`) with `NaN`.
  - Imputes missing sensor measurements using feature-wise median values.
  - Removes duplicate records based on `(source_file, sample_index)`.
- **Subject Demographic Merging:** Joins participant metadata (weight, height, age, gender) using a validated many-to-one left merge on `subject_id`.
- **Outputs:** `motionsense_output/motionsense_initial_clean.csv` and `cleaning_summary.json`.

### C. Outlier Management (`data scripts/outlier_analysis.py`)
- **IQR-Based Detection:** Computes the 25th percentile ($Q_1$), 75th percentile ($Q_3$), and Interquartile Range ($\text{IQR} = Q_3 - Q_1$) across all 12 sensor features.
- **Outlier Capping (Winsorization):** Clamps extreme measurements outside $[Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$ to boundary values to prevent gradient explosion while retaining temporal continuity.
- **Outputs:** `motionsense_output/motionsense_cleaned.csv` and `outlier_summary.json`.

### D. Feature Preprocessing & Scaling (`data scripts/feature_preprocessing.py`)
- **Target Label Encoding:** Uses scikit-learn's `LabelEncoder` to convert categorical activity labels into numeric targets (`0` to `5`).
- **Label Mapping Export:** Saves `motionsense_output/activity_label_mapping.json` for deterministic class decoding during evaluation.
- **Feature Standardization:** Fits a `StandardScaler` ($\mu = 0, \sigma = 1$) across all 12 sensor features:
  $$\tilde{x} = \frac{x - \mu}{\sigma}$$
- **Scaler Persistence:** Serializes the fitted scaler to `motionsense_output/sensor_scaler.pkl` with `joblib` for identical preprocessing of future test streams.
- **Output Dataset:** `motionsense_output/motionsense_preprocessed.csv` containing original columns, encoded labels (`activity_encoded`), and scaled features (`*_scaled`).

### E. Exploratory Data Analysis (`data scripts/eda.py`)
- **Class Balance Analysis:** Generates bar charts showing the sample distribution across all 6 activities.
- **Sensor Distribution Histograms:** Plots frequency distributions for acceleration, gravity, rotation rate, and attitude.
- **Sensor Correlation Heatmap:** Computes and plots a $12 \times 12$ Pearson correlation matrix illustrating relationships between linear acceleration and gravitational components.

### F. Phase 2 Pipeline Runner (`data scripts/run_pipeline.py`)
- **Sequential Orchestrator:** Runs `download_data.py` $\to$ `clean_data.py` $\to$ `outlier_analysis.py` $\to$ `feature_preprocessing.py` $\to$ `eda.py` sequentially with halt-on-failure execution.

---

## 3. New Features: Phase 3 Model Architecture & Implementation (Member 1)

Member 1 designed, implemented, and verified the complete neural network architecture for Human Activity Recognition under reliable and degraded sensor conditions:

### A. Deep Learning Framework Selection & Technical Justification
- **Framework:** **PyTorch** (`torch >= 2.0.0`).
- **Justifications:**
  1. *Define-by-Run Dynamic Graphs:* Natively supports runtime sensor availability masks ($m_i \in \{0, 1\}$) and dynamic reliability gating without static graph compilation constraints.
  2. *Modular Multi-Branch Construction:* Clean encapsulation via `nn.Module` subclasses (`SensorBranchEncoder`, `AdaptiveFusionModule`, `BaselineCNN`, `AdaptiveSensorFusion`).
  3. *Optimized 1D Temporal Convolutions:* High-throughput temporal processing via `nn.Conv1d` and `nn.BatchNorm1d`.
  4. *Window Invariance via Adaptive Pooling:* `nn.AdaptiveAvgPool1d(1)` aggregates temporal features into fixed embeddings, allowing variable window lengths ($L=64, 128, 200, 256$).
  5. *Decoupled Design:* Architecture is strictly separated from Member 2's training pipeline and Member 3's evaluation suite.

### B. Sensor Feature Partitioning (12 Channels $\to$ 2 Physical Domains)
To preserve physical semantics and prevent noisy accelerometer shocks from degrading attitude readings, the 12 MotionSense features are partitioned:

| Index | Feature Name | Physical Modality | Architectural Branch |
| :---: | :--- | :--- | :--- |
| `0` | `attitude.roll` | Orientation (Euler angle) | **Branch 2 (Gyroscope & Attitude)** |
| `1` | `attitude.pitch` | Orientation (Euler angle) | **Branch 2 (Gyroscope & Attitude)** |
| `2` | `attitude.yaw` | Orientation (Euler angle) | **Branch 2 (Gyroscope & Attitude)** |
| `3` | `gravity.x` | Directional Gravity Vector | **Branch 1 (Accelerometer & Gravity)** |
| `4` | `gravity.y` | Directional Gravity Vector | **Branch 1 (Accelerometer & Gravity)** |
| `5` | `gravity.z` | Directional Gravity Vector | **Branch 1 (Accelerometer & Gravity)** |
| `6` | `rotationRate.x` | Angular Velocity | **Branch 2 (Gyroscope & Attitude)** |
| `7` | `rotationRate.y` | Angular Velocity | **Branch 2 (Gyroscope & Attitude)** |
| `8` | `rotationRate.z` | Angular Velocity | **Branch 2 (Gyroscope & Attitude)** |
| `9` | `userAcceleration.x`| Linear Dynamic Acceleration | **Branch 1 (Accelerometer & Gravity)** |
| `10`| `userAcceleration.y`| Linear Dynamic Acceleration | **Branch 1 (Accelerometer & Gravity)** |
| `11`| `userAcceleration.z`| Linear Dynamic Acceleration | **Branch 1 (Accelerometer & Gravity)** |

---

### C. Baseline 1D CNN (`src/model.py`)
- **Class:** `BaselineCNN(in_channels=12, num_classes=6)`
- **Design:** Single-stream 1D CNN processing all 12 sensor channels simultaneously.
- **Layer Architecture:**
  - `Conv1D Block 1`: `Conv1d(12 -> 64, k=5, p=2)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` + `Dropout(0.20)`
  - `Conv1D Block 2`: `Conv1d(64 -> 128, k=5, p=2)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` + `Dropout(0.20)`
  - `Conv1D Block 3`: `Conv1d(128 -> 256, k=3, p=1)` + `BatchNorm1d` + `ReLU` + `AdaptiveAvgPool1d(1)` + `Dropout(0.30)`
  - `Classifier Head`: `Linear(256 -> 128)` + `ReLU` + `Dropout(0.30)` + `Linear(128 -> 6)`
- **Total Trainable Parameters:** **177,670**
- **Smart Input Handling:** Accepts standard PyTorch shape `(batch, 12, length)` as well as automatically permuting `(batch, length, 12)`.
- **Inference Interfaces:**
  - `forward(x)`: Raw classification logits $\in \mathbb{R}^{B \times 6}$ for `CrossEntropyLoss`.
  - `predict(x)`: Predicted class indices $\in \{0, 1, 2, 3, 4, 5\}$.
  - `predict_proba(x)`: Softmax probability distribution summing to 1.0.
- **Convenience Methods:** `get_loss_criterion()` (`CrossEntropyLoss`), `get_optimizer(lr=1e-3)` (`Adam`), `count_parameters()`.
- **Model Checkpointing:** `save_model(model, path)` and `load_model(model, path)`.
- **Self-Testing Routine:** Embedded test suite covering forward pass, shape transposition, softmax normalization, loss/optimizer initialization, and variable window lengths ($L=64, 200, 256$).

---

### D. Adaptive Sensor Fusion Architecture (`src/sensor_fusion.py`)
- **Class:** `AdaptiveSensorFusion`
- **Total Trainable Parameters:** **265,032**
  - Accelerometer Branch Encoder: 109,440
  - Gyroscope Branch Encoder: 109,440
  - Adaptive Fusion Module: 12,482
  - Classification Head: 33,670

#### 1. Specialized Branch Encoders (`SensorBranchEncoder`)
- Independent 1D CNN encoders for Branch 1 (Accelerometer) and Branch 2 (Gyroscope).
- Architecture per branch:
  - `Conv1D(6 -> 64, k=5, p=2)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` + `Dropout(0.20)`
  - `Conv1D(64 -> 128, k=5, p=2)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` + `Dropout(0.20)`
  - `Conv1D(128 -> 128, k=3, p=1)` + `BatchNorm1d` + `ReLU` + `AdaptiveAvgPool1d(1)` + `Dropout(0.20)`
  - `Linear(128 -> 128)` + `LayerNorm(128)` + `ReLU`
- Outputs 128-dimensional latent representations: $\mathbf{h}_{\text{acc}} \in \mathbb{R}^{B \times 128}$ and $\mathbf{h}_{\text{gyro}} \in \mathbb{R}^{B \times 128}$.

#### 2. Adaptive Fusion Mechanism (`AdaptiveFusionModule`)
- **Availability Masks ($m_{\text{acc}}, m_{\text{gyro}} \in \{0, 1\}$):** Explicit indicators of physical sensor availability.
- **Reliability Metrics ($r_{\text{acc}}, r_{\text{gyro}} \in [0, 1]$):** Quality scores indicating signal-to-noise ratio or corruption level. If omitted, internal self-confidence estimators compute confidence directly from latent features.
- **Effective Reliability:**
  $$R_{\text{acc}} = m_{\text{acc}} \cdot r_{\text{acc}}, \qquad R_{\text{gyro}} = m_{\text{gyro}} \cdot r_{\text{gyro}}$$
- **Content Importance Scoring:** $s_i = G(\mathbf{h}_i)$ via a 2-layer MLP gating projection.
- **Dynamic Weight Calculation:**
  $$w_i = \frac{R_i \cdot \exp(s_i - s_{\max})}{\sum_k R_k \cdot \exp(s_k - s_{\max}) + \epsilon}$$
- **Division-by-Zero Guard:** $\epsilon = 10^{-8}$ prevents numerical instability even when both sensors are completely disconnected or report zero reliability.
- **Boundary Guarantees:**
  - $m_{\text{acc}} = 0 \implies w_{\text{acc}} = 0.0, \; w_{\text{gyro}} = 1.0$ (100% reliance on Gyroscope/Attitude).
  - $m_{\text{gyro}} = 0 \implies w_{\text{gyro}} = 0.0, \; w_{\text{acc}} = 1.0$ (100% reliance on Accelerometer/Gravity).
  - $m_{\text{acc}} = 0 \text{ and } m_{\text{gyro}} = 0 \implies w_{\text{acc}} = 0.5, \; w_{\text{gyro}} = 0.5$ (neutral fallback).
- **Corrupted Signal Leakage Prevention:** Multiplies latent embeddings by availability masks: $\mathbf{h}_i^{\text{eff}} = \mathbf{h}_i \cdot m_i$ to ensure noise or invalid readings are zeroed out before reaching the classifier.
- **Dual-Branch Fused Embedding:**
  $$\mathbf{h}_{\text{fused}} = \left[ w_{\text{acc}} \cdot \mathbf{h}_{\text{acc}}^{\text{eff}} \;\|\; w_{\text{gyro}} \cdot \mathbf{h}_{\text{gyro}}^{\text{eff}} \right] \in \mathbb{R}^{B \times 256}$$

#### 3. Dual Input Interfaces
- **Unified Mode:** `model(x)` where `x` is the standard 12-channel tensor (automatically sliced into Branch 1 and Branch 2).
- **Separate Mode:** `model(x_acc=x_acc, x_gyro=x_gyro)` where each branch receives its dedicated 6-channel stream directly.

#### 4. Verification Suite (7 Comprehensive Tests in `src/sensor_fusion.py`)
1. *Normal Condition:* Both available $\implies$ both branches contribute ($w_{\text{acc}} \approx 0.52, w_{\text{gyro}} \approx 0.48$).
2. *Accelerometer Missing:* $m_{\text{acc}}=0 \implies w_{\text{acc}}=0.0, w_{\text{gyro}}=1.0$.
3. *Gyroscope Missing:* $m_{\text{gyro}}=0 \implies w_{\text{acc}}=1.0, w_{\text{gyro}}=0.0$.
4. *Unequal Reliability:* $r_{\text{acc}}=0.15, r_{\text{gyro}}=0.95 \implies w_{\text{acc}}=0.14, w_{\text{gyro}}=0.86$ (dynamically favors cleaner sensor).
5. *Separate Branch Inputs:* `x_acc` and `x_gyro` processed without error.
6. *Numerical Stability & Edge Cases:* Extreme all-zero input with 0.0 reliability verified; division-by-zero safely avoided; finite outputs guaranteed.
7. *Inference & Optimizer:* Softmax normalization, `CrossEntropyLoss`, `Adam`, and variable window lengths verified.

---

### E. Quick Integration & Verification Script (`test_models.py`)
- A standalone Python script in the repository root allowing teammates and reviewers to verify both models with a single command:
  ```bash
  python test_models.py
  ```
- Tests input window creation, baseline prediction, adaptive forward pass, dynamic weight inspection, and missing sensor simulation.

---

### F. High-Resolution Architecture Diagram (`diagrams/model_architecture.png`)
- **Generator Script:** `src/generate_diagram.py`.
- **Output:** 300 DPI publication-quality visual diagram comparing:
  - Left panel: Single-stream Baseline 1D CNN with parameter counts and layer shapes.
  - Right panel: Dual-branch Adaptive Sensor Fusion model showcasing the specialized branch encoders, availability masks, reliability metrics, adaptive gating formulas, and classification head.

---

### G. LaTeX Report Section (`Reports/model_architecture.tex`)
- Academic-standard LaTeX document ready to be included (`\input{Reports/model_architecture.tex}`) into the group's final report.
- Fully formulated with:
  - Deep learning framework selection and justification.
  - Baseline CNN layer descriptions and parameter accounting.
  - Mathematical equations for availability masks, effective reliability, dynamic gating, and boundary constraints.
  - Loss function ($\mathcal{L}_{\text{CE}}$) and optimizer specifications ($\text{Adam}$).
  - Theoretical justification for how adaptive fusion prevents failure under real-world sensor degradation.

---

### H. Packaging & Dependency Files
- **`src/__init__.py`:** Exposes `BaselineCNN`, `AdaptiveSensorFusion`, `SensorBranchEncoder`, `AdaptiveFusionModule`, `save_model`, `load_model`, and channel constants (`ACC_INDICES`, `GYRO_INDICES`, `SENSOR_COLUMNS`).
- **`requirements.txt`:** Specifies framework versions (`torch>=2.0.0`, `matplotlib>=3.7.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`).

---

## 4. Repository Structure Comparison

```text
======================= BEFORE (Phase 2) =======================
adaptive-sensor-fusion/
├── README.md
├── .gitignore
├── adaptive_sensor_fusion.ipynb
├── data scripts/
│   ├── download_data.py
│   ├── clean_data.py
│   ├── outlier_analysis.py
│   ├── feature_preprocessing.py
│   ├── eda.py
│   └── run_pipeline.py
├── Reports/
│   └── Adaptive_sensor_Report .pdf
└── motionsense_output/ (generated locally)

======================= AFTER (Phase 3 Added) =======================
adaptive-sensor-fusion/
├── README.md                           # Updated with Phase 3 instructions & team roles
├── .gitignore
├── requirements.txt                    # [NEW] PyTorch & project dependencies
├── test_models.py                      # [NEW] Quick model verification script
├── adaptive_sensor_fusion.ipynb
├── data scripts/                       # [UNMODIFIED] Preserved Phase 2 scripts
│   ├── download_data.py
│   ├── clean_data.py
│   ├── outlier_analysis.py
│   ├── feature_preprocessing.py
│   ├── eda.py
│   └── run_pipeline.py
├── src/                                # [NEW] Core Phase 3 deep learning package
│   ├── __init__.py                     # [NEW] Package exports
│   ├── model.py                        # [NEW] Baseline 1D CNN (177,670 params)
│   ├── sensor_fusion.py                # [NEW] Adaptive Sensor Fusion (265,032 params)
│   └── generate_diagram.py             # [NEW] Architecture diagram generator
├── diagrams/                           # [NEW] Visual figures
│   └── model_architecture.png          # [NEW] 300 DPI architecture diagram
├── Reports/
│   ├── Adaptive_sensor_Report .pdf     # [UNMODIFIED] Existing Phase 2 report
│   └── model_architecture.tex          # [NEW] Member 1 LaTeX report section
└── motionsense_output/ (generated locally)
```

---

## 5. Team Integration Guide

| Team Member | Role | Integration Points with Phase 3 Code |
| :--- | :--- | :--- |
| **Member 1 (You)** | **Architecture & Implementation** | Built `src/model.py`, `src/sensor_fusion.py`, `diagrams/model_architecture.png`, `Reports/model_architecture.tex`. |
| **Member 2** | **Training Pipeline** | - Feeds `(batch_size, 12, sequence_length)` preprocessed windows.<br>- Uses `model.get_loss_criterion()` (`CrossEntropyLoss`).<br>- Uses `model.get_optimizer(lr=1e-3)` (`Adam`).<br>- Saves checkpoints with `save_model()` and `save_adaptive_model()`. |
| **Member 3** | **Evaluation & Robustness** | - Uses `model.predict(x)` and `model.predict_proba(x)`.<br>- Simulates sensor dropout via `mask_acc=0.0` or `mask_gyro=0.0`.<br>- Simulates sensor noise via `reliability_acc=0.2, reliability_gyro=0.9`.<br>- Inspects dynamic weights via `logits, weights = model(x, return_weights=True)`. |

---

## 6. How to Run Everything (Quick Reference)

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the quick end-to-end verification demo
python test_models.py

# 3. Run comprehensive tests on the Baseline CNN
python src/model.py

# 4. Run comprehensive tests on the Adaptive Sensor Fusion model
python src/sensor_fusion.py

# 5. Re-generate the publication-quality architecture diagram
python src/generate_diagram.py
```

