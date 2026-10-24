import os
import re
import json
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configurations & Constants
DATA_URL = 'https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip'
SUBJECT_URL = 'https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/data_subjects_info.csv'

ACTIVITY_LABELS = {
    'dws': 'downstairs',
    'ups': 'upstairs',
    'wlk': 'walking',
    'jog': 'jogging',
    'std': 'standing',
    'sit': 'sitting'
}

SENSOR_COLUMNS = [
    'attitude.roll', 'attitude.pitch', 'attitude.yaw',
    'gravity.x', 'gravity.y', 'gravity.z',
    'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
    'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'
]

def run_pipeline():
    print("Starting end-to-end Sensor Fusion pipeline...")
    
    # Setup Output Directory
    script_path = Path(__file__).resolve()
    project_dir = script_path.parent.parent
    output_dir = project_dir / 'motionsense_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Datasets
    download_time = datetime.now(timezone.utc).isoformat()
    print("Downloading sensor data and subject metadata...")
    
    resp = requests.get(DATA_URL, timeout=120)
    resp.raise_for_status()
    zip_bytes = resp.content
    zip_file = zipfile.ZipFile(BytesIO(zip_bytes))
    
    sub_resp = requests.get(SUBJECT_URL, timeout=60)
    sub_resp.raise_for_status()
    subjects = pd.read_csv(BytesIO(sub_resp.content))
    subjects.columns = subjects.columns.str.strip()
    
    # Generate and Save Collection Report
    collection_report = {
        "dataset": "MotionSense",
        "sensor_data_url": DATA_URL,
        "participant_data_url": SUBJECT_URL,
        "download_time_utc": download_time,
        "sensor_data_http_status": resp.status_code,
        "participant_data_http_status": sub_resp.status_code,
        "sensor_zip_size_bytes": len(zip_bytes),
        "zip_file_count": len(zip_file.namelist()),
        "participant_records_downloaded": len(subjects)
    }
    with open(output_dir / 'collection_report.json', 'w', encoding='utf-8') as f:
        json.dump(collection_report, f, indent=4)
        
    # 2. Extract and Parse Archive Files
    print("Parsing ZIP file entries...")
    all_data = []
    for file in zip_file.namelist():
        if not file.endswith('.csv') or 'A_DeviceMotion_data/' not in file:
            continue
            
        file_name = Path(file).name
        if file_name.startswith('._') or '__MACOSX' in file:
            continue
            
        match = re.fullmatch(r'sub_(\d+)\.csv', file_name)
        if match is None:
            continue
            
        subject_id = int(match.group(1))
        folder_parts = Path(file).parent.name.split('_')
        if len(folder_parts) < 2:
            continue
            
        activity_code = folder_parts[0]
        if activity_code not in ACTIVITY_LABELS:
            continue
            
        try:
            trial = int(folder_parts[-1])
        except ValueError:
            continue
            
        df = pd.read_csv(zip_file.open(file))
        df.columns = df.columns.str.strip()
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'sample_index'})
            
        df['activity'] = ACTIVITY_LABELS[activity_code]
        df['activity_code'] = activity_code
        df['subject_id'] = subject_id
        df['trial'] = trial
        df['source_file'] = file
        all_data.append(df)
        
    raw_df = pd.concat(all_data, ignore_index=True)
    
    # 3. Clean and Impute
    print("Cleaning data, aligning subjects, and capping outliers...")
    data = raw_df.copy()
    for col in SENSOR_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data[SENSOR_COLUMNS] = data[SENSOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
    
    for col in SENSOR_COLUMNS:
        median_val = data[col].median()
        if pd.notna(median_val):
            data[col] = data[col].fillna(median_val)
            
    if 'sample_index' in data.columns:
        data = data.drop_duplicates(subset=['source_file', 'sample_index'])
    else:
        data = data.drop_duplicates()
        
    total_capped = 0
    for col in SENSOR_COLUMNS:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        total_capped += int(((data[col] < lower) | (data[col] > upper)).sum())
        data[col] = data[col].clip(lower=lower, upper=upper)
        
    # Merge subject metadata
    subjects_df = subjects.rename(columns={'code': 'subject_id'}).copy() if 'code' in subjects.columns else subjects.copy()
    data['subject_id'] = pd.to_numeric(data['subject_id'], errors='coerce').astype('Int64')
    subjects_df['subject_id'] = pd.to_numeric(subjects_df['subject_id'], errors='coerce').astype('Int64')
    subjects_df = subjects_df.dropna(subset=['subject_id']).drop_duplicates(subset=['subject_id'])
    
    data = data.merge(subjects_df, on='subject_id', how='left')
    
    cleaning_summary = {
        "raw_rows": len(raw_df),
        "final_rows": len(data),
        "outlier_values_detected_and_capped": total_capped
    }
    with open(output_dir / 'cleaning_summary.json', 'w', encoding='utf-8') as f:
        json.dump(cleaning_summary, f, indent=4)
        
    # Save Cleaned Dataset
    cleaned_file_path = output_dir / 'motionsense_cleaned.csv'
    data.to_csv(cleaned_file_path, index=False)
    
    # 4. Preprocess (Scale & Encode Labels)
    print("Standardizing features and encoding targets...")
    le = LabelEncoder()
    data['activity_encoded'] = le.fit_transform(data['activity'].astype(str))
    
    label_map = {str(lbl): int(code) for lbl, code in zip(le.classes_, le.transform(le.classes_))}
    with open(output_dir / 'activity_label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4)
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[SENSOR_COLUMNS])
    scaled_cols = [f"{c}_scaled" for c in SENSOR_COLUMNS]
    data[scaled_cols] = scaled_data
    
    preprocessed_file_path = output_dir / 'motionsense_preprocessed.csv'
    data.to_csv(preprocessed_file_path, index=False)
    
    # 5. Generate Visual Charts
    print("Generating analysis visualizations...")
    
    # Class Distribution
    plt.figure(figsize=(9, 5))
    sns.countplot(data=data, x='activity', order=data['activity'].value_counts().index)
    plt.title('MotionSense Activity Class Distribution')
    plt.xlabel('Activity')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / 'activity_distribution.png', dpi=300)
    plt.close()
    
    # Correlation Matrix
    plt.figure(figsize=(12, 9))
    sns.heatmap(data[SENSOR_COLUMNS].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Between Sensor Features')
    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_correlation_heatmap.png', dpi=300)
    plt.close()
    
    # Distributions Histogram
    data[SENSOR_COLUMNS[:6]].hist(figsize=(12, 8), bins=30)
    plt.suptitle('Distributions of Selected Sensor Features', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_feature_distributions.png', dpi=300)
    plt.close()
    
    print("End-to-end processing pipeline complete!")

if __name__ == '__main__':
    run_pipeline()





















































###############################################


# --------------------------------------------------
# 1. Download the MotionSense dataset
# --------------------------------------------------

data_url = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"

subject_url = "https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/data_subjects_info.csv"

response = requests.get(data_url, timeout=120)
response.raise_for_status()

zip_file = zipfile.ZipFile(BytesIO(response.content))

print("Dataset downloaded successfully.")

# --------------------------------------------------
# 2. Define activity labels and sensor columns
# --------------------------------------------------

activity_labels = {
    "dws": "downstairs",
    "ups": "upstairs",
    "wlk": "walking",
    "jog": "jogging",
    "std": "standing",
    "sit": "sitting"
}

sensor_columns = [
    "attitude.roll",
    "attitude.pitch",
    "attitude.yaw",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z"
]

# --------------------------------------------------
# 3. Read and combine valid sensor CSV files
# --------------------------------------------------

all_data = []
skipped_files = 0

for file in zip_file.namelist():

    # Only process CSV files inside the sensor data folder
    if not file.endswith(".csv"):
        continue

    if "A_DeviceMotion_data/" not in file:
        continue

    file_name = Path(file).name

    # Skip macOS metadata files such as ._13.csv
    if file_name.startswith("._") or "__MACOSX" in file:
        skipped_files += 1
        continue

    # Accept only filenames such as sub_13.csv
    match = re.fullmatch(r"sub_(\d+)\.csv", file_name)

    if match is None:
        skipped_files += 1
        continue

    subject_id = int(match.group(1))
    folder_name = Path(file).parent.name

    # Extract activity and trial number
    folder_parts = folder_name.split("_")

    if len(folder_parts) < 2:
        skipped_files += 1
        continue

    activity_code = folder_parts[0]

    if activity_code not in activity_labels:
        skipped_files += 1
        continue

    try:
        trial = int(folder_parts[-1])
    except ValueError:
        skipped_files += 1
        continue

    # Read CSV file
    df = pd.read_csv(zip_file.open(file))

    # Rename unnecessary index column
    if "Unnamed: 0" in df.columns:
        df = df.rename(
            columns={"Unnamed: 0": "sample_index"}
        )

    # Check whether required sensor columns exist
    if not all(col in df.columns for col in sensor_columns):
        print("Skipping file with missing columns:", file)
        skipped_files += 1
        continue

    # Add activity and participant information
    df["activity"] = activity_labels[activity_code]
    df["activity_code"] = activity_code
    df["subject_id"] = subject_id
    df["trial"] = trial
    df["source_file"] = file

    all_data.append(df)

# Combine all valid files
if len(all_data) == 0:
    raise ValueError(
        "No valid sensor CSV files found. "
        "Check the ZIP file structure."
    )

data = pd.concat(all_data, ignore_index=True)

print("Valid files loaded:", len(all_data))
print("Files skipped:", skipped_files)
print("Combined dataset shape:", data.shape)

# --------------------------------------------------
# 4. Standardize column names
# --------------------------------------------------

data.columns = data.columns.str.strip()

# --------------------------------------------------
# 5. Convert sensor columns to numeric
# --------------------------------------------------

for col in sensor_columns:
    data[col] = pd.to_numeric(
        data[col], errors="coerce"
    )

# Replace infinite values with NaN
data[sensor_columns] = data[sensor_columns].replace(
    [np.inf, -np.inf], np.nan
)

print("\nMissing sensor values before cleaning:")
print(data[sensor_columns].isnull().sum())

# --------------------------------------------------
# 6. Remove rows with missing sensor values
# --------------------------------------------------

rows_before = len(data)

data = data.dropna(
    subset=sensor_columns
).copy()

print(
    "\nRows removed due to missing values:",
    rows_before - len(data)
)

# --------------------------------------------------
# 7. Remove duplicate samples
# --------------------------------------------------

rows_before = len(data)

if "sample_index" in data.columns:
    data = data.drop_duplicates(
        subset=["source_file", "sample_index"]
    )
else:
    data = data.drop_duplicates()

print(
    "Duplicate rows removed:",
    rows_before - len(data)
)

data = data.reset_index(drop=True)

# --------------------------------------------------
# 8. Download participant information
# --------------------------------------------------

subject_response = requests.get(
    subject_url, timeout=60
)

subject_response.raise_for_status()

subjects = pd.read_csv(
    BytesIO(subject_response.content)
)

subjects.columns = subjects.columns.str.strip()

# Rename participant ID column
if "code" in subjects.columns:
    subjects = subjects.rename(
        columns={"code": "subject_id"}
    )
elif "subject_id" not in subjects.columns:
    raise ValueError(
        "Participant ID column not found."
    )

# Convert participant IDs to numeric integers
data["subject_id"] = pd.to_numeric(
    data["subject_id"], errors="coerce"
).astype("Int64")

subjects["subject_id"] = pd.to_numeric(
    subjects["subject_id"], errors="coerce"
).astype("Int64")

# Remove duplicate participant records
subjects = subjects.drop_duplicates(
    subset=["subject_id"]
)

# Merge participant details into the main dataset
data = data.merge(
    subjects,
    on="subject_id",
    how="left",
    validate="many_to_one",
    suffixes=("", "_subject")
)

print(
    "\nShape after merging participant information:",
    data.shape
)

# --------------------------------------------------
# 9. Validate the cleaned dataset
# --------------------------------------------------

print("\n========== CLEANING SUMMARY ==========")

print("Final dataset shape:", data.shape)

print(
    "Number of sensor features:",
    len(sensor_columns)
)

print(
    "Number of activities:",
    data["activity"].nunique()
)

print(
    "Activities:",
    sorted(data["activity"].dropna().unique())
)

print(
    "Number of participants:",
    data["subject_id"].nunique()
)

print("\nActivity distribution:")
print(data["activity"].value_counts())

print("\nMissing values:")
missing_values = data.isnull().sum()

print(
    missing_values[missing_values > 0]
)

print(
    "\nInfinite sensor values:",
    np.isinf(data[sensor_columns].to_numpy()).sum()
)

if "sample_index" in data.columns:
    print(
        "\nDuplicate file/sample-index pairs:",
        data.duplicated(
            subset=["source_file", "sample_index"]
        ).sum()
    )

print("\nSensor statistics:")
display(data[sensor_columns].describe())

# --------------------------------------------------
# 10. Save the cleaned dataset
# --------------------------------------------------

output_file = "motionsense_cleaned.csv"

data.to_csv(output_file, index=False)

print(
    "\nCleaned dataset saved successfully as:",
    output_file
)

print("Final rows:", len(data))
print("Final columns:", len(data.columns))

# Display first five rows
display(data.head())