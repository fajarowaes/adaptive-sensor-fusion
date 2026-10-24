# Adaptive Sensor Fusion

## Project Overview

Adaptive Sensor Fusion is a data science project that works with motion sensor measurements to support human activity recognition. It uses the MotionSense dataset, collected from smartphone motion sensors, to prepare structured data for further analysis and machine learning.

## Dataset

The project uses the MotionSense dataset.

- **Sensor data:** [MotionSense Device Motion Data](https://github.com/mmalekzadeh/motion-sense)
- **Participant information:** [Data Subjects Information](https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/data_subjects_info.csv)

The dataset contains smartphone motion measurements associated with activities such as walking, jogging, sitting, standing, and moving upstairs or downstairs.

## Phase 2: Data Acquisition and Preprocessing

The Phase 2 pipeline performs the following tasks:

1. **Data collection:** Downloads sensor data and participant metadata from the dataset's public repository.
2. **Data structuring:** Combines sensor CSV files and adds activity labels, participant IDs, trial numbers, and source-file information.
3. **Data cleaning:** Converts sensor measurements to numeric values, handles missing and infinite values, and removes duplicate samples.
4. **Outlier management:** Uses the Interquartile Range (IQR) method to detect and cap extreme sensor measurements.
5. **Participant metadata integration:** Merges participant information with the sensor data.
6. **Feature preparation:** Standardizes sensor features and encodes activity labels.
7. **Exploratory Data Analysis:** Generates activity distribution charts, sensor feature distribution histograms, and a sensor correlation heatmap.

## Project Structure

- `adaptive_sensor_fusion.ipynb`: Interactive notebook for analysis and visualization.
- `data scripts/acquisition_and_cleaning.py`: Main Phase 2 data acquisition and preprocessing script.
- `.gitignore`: Excludes generated datasets and temporary files.
- `motionsense_output/`: Locally generated reports, cleaned data, and visualizations.

## Requirements

Python 3.10 or a compatible Python version, with the following packages:

- pandas
- numpy
- requests
- matplotlib
- seaborn
- scikit-learn

## How to Run

Install the required packages:

```bash
python -m pip install pandas numpy requests matplotlib seaborn scikit-learn
```

Run the Phase 2 pipeline from the repository root:

```bash
python "data scripts/acquisition_and_cleaning.py"
```

The script downloads the dataset, processes the sensor records, and generates output files in `motionsense_output/`.

## Data and Repository Policy

Raw datasets and large generated CSV files are excluded from GitHub using `.gitignore`. The preprocessing code and project documentation are maintained in the repository so the pipeline can be reproduced.

## Team Contributions

Add the names of all group members and briefly describe each member's actual contributions here.
