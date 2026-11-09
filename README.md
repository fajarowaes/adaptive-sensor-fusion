# Adaptive Sensor Fusion

## Project Overview

Adaptive Sensor Fusion is a data science project focused on Human Activity Recognition (HAR) using smartphone motion sensor measurements. The project uses the MotionSense dataset to prepare sensor data for analysis and machine learning, with the broader goal of recognizing human activities and investigating how activity recognition can remain reliable when sensor measurements become noisy or unavailable.

## Dataset

The project uses the publicly available MotionSense dataset.

- **Sensor Data:** [MotionSense Device Motion Data](https://github.com/mmalekzadeh/motion-sense)
- **Participant Information:** [Data Subjects Information](https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/data_subjects_info.csv)

The dataset contains smartphone motion measurements associated with six activities: downstairs, upstairs, walking, jogging, standing, and sitting.

The sensor features include device attitude, gravity, rotation rate, and user acceleration measurements.

## Phase 2: Data Acquisition and Preprocessing

The Phase 2 pipeline automates data collection, cleaning, preprocessing, and exploratory data analysis.

1. **Data Acquisition:** Downloads the MotionSense sensor dataset and participant metadata.
2. **Data Structuring:** Combines sensor CSV files and adds activity labels, participant IDs, trial numbers, and source-file information.
3. **Data Cleaning:** Converts sensor measurements to numeric values, handles missing and infinite values, removes duplicate samples, and integrates participant metadata.
4. **Outlier Management:** Uses the Interquartile Range (IQR) method to identify and cap extreme sensor measurements.
5. **Feature Preprocessing:** Encodes activity labels and standardizes all 12 sensor features using `StandardScaler`.
6. **Scaler Saving:** Saves the fitted scaler for consistent transformation of future sensor measurements.
7. **Exploratory Data Analysis:** Generates activity distribution charts, sensor feature distribution histograms, and a sensor correlation heatmap.

The pipeline processes approximately 1.4 million sensor records across 364 sensor CSV files and 24 participants.

## Project Structure

```text
adaptive-sensor-fusion/
├── README.md
├── .gitignore
├── requirements.txt
├── adaptive_sensor_fusion.ipynb
├── data scripts/
│   ├── download_data.py
│   ├── clean_data.py
│   ├── outlier_analysis.py
│   ├── feature_preprocessing.py
│   ├── eda.py
│   └── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── model.py              # Member 1: Baseline 1D CNN
│   ├── sensor_fusion.py      # Member 1: Adaptive Sensor Fusion Model
│   └── generate_diagram.py   # Architecture diagram generator
├── diagrams/
│   └── model_architecture.png # Model architecture diagram
├── Reports/
│   ├── Adaptive_sensor_Report .pdf
│   └── model_architecture.tex # Member 1: LaTeX Model Architecture Report
└── motionsense_output/ # Generated locally; ignored by Git
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

The `motionsense_output/` directory is generated locally when the pipeline runs. Generated datasets, reports, and visualizations are not required to be committed to GitHub.

## Requirements

- Python 3.10 or a compatible Python version
- pandas
- numpy
- requests
- matplotlib
- seaborn
- scikit-learn
- joblib

Install the required packages:

```bash
python -m pip install pandas numpy requests matplotlib seaborn scikit-learn joblib
```

## How to Run

Run the complete Phase 2 pipeline from the repository root:

```bash
python "data scripts/run_pipeline.py"
```

The pipeline executes the scripts in this order:

1. `download_data.py`
2. `clean_data.py`
3. `outlier_analysis.py`
4. `feature_preprocessing.py`
5. `eda.py`

The scripts must remain in the `data scripts/` directory. The pipeline stops if any script fails.

## Main Outputs

- **`motionsense_raw.zip`:** Downloaded raw sensor dataset.
- **`data_subjects_info.csv`:** Participant metadata.
- **`motionsense_cleaned.csv`:** Cleaned sensor dataset with outlier-capped measurements.
- **`motionsense_preprocessed.csv`:** Dataset containing encoded activity labels and scaled sensor features.
- **`sensor_scaler.pkl`:** Saved fitted scaler.
- **`activity_label_mapping.json`:** Mapping between activity names and encoded labels.
- **JSON reports:** Summaries of data collection, cleaning, and outlier analysis.
- **EDA charts:** Activity distribution, sensor feature distributions, and sensor correlation heatmap.

## Phase 3: Adaptive Sensor Fusion and Human Activity Recognition

Phase 3 implements deep learning models for Human Activity Recognition under reliable and degraded sensor conditions.

### Member 1: Model Architecture and Implementation (Completed)

Member 1 is responsible for designing, implementing, and verifying the core neural network architectures:

1. **`src/model.py` (`BaselineCNN`):**
   - Single-stream 1D Convolutional Neural Network processing all 12 MotionSense sensor features.
   - Three 1D convolutional blocks with Batch Normalization, ReLU, Max-Pooling, Spatial Dropout, and Adaptive Average Pooling (`AdaptiveAvgPool1d(1)`).
   - Fully connected classification head outputting 6 activity logits.
   - Total trainable parameters: **177,670**.

2. **`src/sensor_fusion.py` (`AdaptiveSensorFusion`):**
   - Dual-branch architecture decoupling **Branch 1 (Accelerometer & Gravity: 6 channels)** and **Branch 2 (Gyroscope & Attitude: 6 channels)**.
   - Dedicated branch feature encoders (`SensorBranchEncoder`) producing 128-dimensional latent representations.
   - `AdaptiveFusionModule` computing dynamic reliability-guided fusion weights:
     $$w_i = \frac{R_i \cdot \exp(s_i)}{\sum_k R_k \cdot \exp(s_k) + \epsilon}$$
   - Strict availability masking ($m_{\text{acc}}, m_{\text{gyro}} \in \{0, 1\}$) ensuring zero leakage when a sensor modality is disconnected ($m_{\text{acc}}=0 \implies w_{\text{acc}}=0, w_{\text{gyro}}=1$).
   - Numerical stability guard ($\epsilon = 10^{-8}$) preventing division-by-zero under total sensor failure.
   - Total trainable parameters: **265,032**.

3. **`diagrams/model_architecture.png`:**
   - Publication-quality comparative architecture diagram generated by `src/generate_diagram.py`.

4. **`Reports/model_architecture.tex`:**
   - Complete, LaTeX-ready report section detailing the framework justification, mathematical formulations, layer descriptions, loss function, optimizer, and reliability analysis.

### How to Run Model Tests

Run the model verification routines from the repository root:

```bash
# Test the Baseline CNN
python src/model.py

# Test the Adaptive Sensor Fusion model
python src/sensor_fusion.py

# Regenerate the architecture diagram
python src/generate_diagram.py
```

### Team Integration Guide

- **For Member 2 (Training Pipeline):**
  - Expected input tensor: `(batch_size, 12, sequence_length)` (e.g., sequence length $L = 128$).
  - `model.get_loss_criterion()` provides `nn.CrossEntropyLoss()`.
  - `model.get_optimizer(learning_rate=1e-3)` provides the pre-configured `Adam` optimizer.
  - Save trained weights with `save_model(model, "models/baseline.pth")` or `save_adaptive_model(model, "models/adaptive.pth")`.

- **For Member 3 (Evaluation & Stress-Testing):**
  - Use `model.predict(x)` for predicted class labels (`0-5`).
  - Use `model.predict_proba(x)` for softmax probability distributions.
  - Test sensor degradation via availability masks: `mask_acc=0.0` or `mask_gyro=0.0`.
  - Test continuous sensor noise via reliability scores: `reliability_acc=0.2, reliability_gyro=0.9`.
  - Inspect dynamic weights during inference: `logits, weights = model(x, return_weights=True)`.

## Data and Repository Policy

Raw datasets and large generated files are excluded from GitHub through `.gitignore`. The repository maintains the preprocessing scripts, notebook, and documentation required to reproduce the workflow.

## Team Contributions

- **Member 1 (Architecture & Implementation):** Designed and implemented the PyTorch Baseline 1D CNN (`src/model.py`) and the dual-branch Adaptive Sensor Fusion model (`src/sensor_fusion.py`) with dynamic reliability gating and sensor availability masking. Created the architecture diagram (`diagrams/model_architecture.png`), requirements, and authored the Phase 3 LaTeX report section (`Reports/model_architecture.tex`).
- **Member 2 (Training & Validation Pipeline):** (Responsible for data windowing, training loops, validation curves, and checkpoint training).
- **Member 3 (Evaluation & Robustness Benchmark):** (Responsible for performance metrics, confusion matrices, and noisy/unavailable sensor stress-tests).
