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
├── adaptive_sensor_fusion.ipynb
├── data scripts/
│   ├── download_data.py
│   ├── clean_data.py
│   ├── outlier_analysis.py
│   ├── feature_preprocessing.py
│   ├── eda.py
│   └── run_pipeline.py
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

The next phase will use the preprocessed dataset to develop and evaluate activity recognition models.

Planned tasks include:

1. Building a baseline human activity recognition model.
2. Simulating noisy or unavailable sensor groups.
3. Developing an adaptive sensor-fusion approach that adjusts to sensor reliability.
4. Comparing model performance under normal and degraded sensor conditions.
5. Evaluating results using accuracy, precision, recall, and F1-score.

These experiments will investigate whether adaptive sensor fusion can improve the reliability of activity recognition when sensor measurements are degraded.

## Data and Repository Policy

Raw datasets and large generated files are excluded from GitHub through `.gitignore`. The repository maintains the preprocessing scripts, notebook, and documentation required to reproduce the workflow.

## Team Contributions

Add all group members' names and briefly describe their actual contributions to the project.
