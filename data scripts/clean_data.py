
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Set the project and output folder paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "motionsense_output"

# Map activity codes to readable activity names
ACTIVITY_LABELS = {
    "dws": "downstairs",
    "ups": "upstairs",
    "wlk": "walking",
    "jog": "jogging",
    "std": "standing",
    "sit": "sitting"
}

# List the 12 sensor features expected in the dataset
SENSOR_COLUMNS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y",
    "userAcceleration.z"
]


def clean_data():
    # Set paths for the downloaded dataset and participant information
    zip_path = OUTPUT_DIR / "motionsense_raw.zip"
    subjects_path = OUTPUT_DIR / "data_subjects_info.csv"

    # Stop if the required input files are missing
    if not zip_path.exists() or not subjects_path.exists():
        raise FileNotFoundError(
            "Run download_data.py before running clean_data.py."
        )

    all_data = []

    print("Reading sensor files...")

    # Open the ZIP file without extracting it to disk
    with zipfile.ZipFile(zip_path) as archive:

        for file in archive.namelist():

            # Process only CSV files inside the sensor data folder
            if not file.endswith(".csv"):
                continue

            if "A_DeviceMotion_data/" not in file:
                continue

            # Skip unwanted macOS metadata files
            filename = Path(file).name

            if filename.startswith("._") or "__MACOSX" in file:
                continue

            # Extract the participant ID from the filename
            match = re.fullmatch(r"sub_(\d+)\.csv", filename)

            if match is None:
                continue

            # Read the activity and trial from the folder name
            folder_parts = Path(file).parent.name.split("_")

            if len(folder_parts) < 2:
                continue

            activity_code = folder_parts[0]

            # Skip folders that do not match known activities
            if activity_code not in ACTIVITY_LABELS:
                continue

            try:
                trial = int(folder_parts[-1])
            except ValueError:
                continue

            # Read the sensor data from the CSV file
            with archive.open(file) as sensor_file:
                df = pd.read_csv(sensor_file)

            df.columns = df.columns.str.strip()

            # Rename the original sample index column
            if "Unnamed: 0" in df.columns:
                df = df.rename(
                    columns={"Unnamed: 0": "sample_index"}
                )

            # Check that all required sensor features exist
            missing_columns = [
                col for col in SENSOR_COLUMNS
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing sensor columns in {file}: "
                    f"{missing_columns}"
                )

            # Add labels and identifiers to each sensor sample
            df["activity"] = ACTIVITY_LABELS[activity_code]
            df["activity_code"] = activity_code
            df["subject_id"] = int(match.group(1))
            df["trial"] = trial
            df["source_file"] = file

            all_data.append(df)

    # Stop if no valid sensor files were found
    if not all_data:
        raise ValueError(
            "No valid sensor CSV files were found in the ZIP."
        )

    # Combine all sensor files into one dataset
    data = pd.concat(all_data, ignore_index=True)
    raw_rows = len(data)

    print("Handling missing values and duplicates...")

    # Convert sensor measurements to numeric values
    for col in SENSOR_COLUMNS:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    # Replace infinite sensor values with missing values
    data[SENSOR_COLUMNS] = data[SENSOR_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    # Count missing sensor values before imputation
    missing_before = {
        col: int(data[col].isna().sum())
        for col in SENSOR_COLUMNS
    }

    # Fill missing sensor values using each feature's median
    for col in SENSOR_COLUMNS:
        median = data[col].median()

        if pd.isna(median):
            raise ValueError(
                f"Cannot impute {col}: the entire column is invalid."
            )

        data[col] = data[col].fillna(median)

    # Count sensor values still missing after imputation
    remaining_missing = int(
        data[SENSOR_COLUMNS].isna().sum().sum()
    )

    # Remove repeated samples within the same source file
    if "sample_index" in data.columns:
        data = data.drop_duplicates(
            subset=["source_file", "sample_index"]
        )
    else:
        data = data.drop_duplicates(
            subset=SENSOR_COLUMNS
            + ["activity", "subject_id", "trial", "source_file"]
        )

    # Record how many rows were removed
    duplicates_removed = raw_rows - len(data)
    rows_after_cleaning = len(data)

    # Load participant information
    subjects = pd.read_csv(subjects_path)
    subjects.columns = subjects.columns.str.strip()

    # Rename the participant ID column to match the sensor data
    if "code" in subjects.columns:
        subjects = subjects.rename(
            columns={"code": "subject_id"}
        )

    if "subject_id" not in subjects.columns:
        raise ValueError(
            "Participant metadata has no subject ID column."
        )

    # Convert participant IDs to the same numeric format
    data["subject_id"] = pd.to_numeric(
        data["subject_id"], errors="coerce"
    ).astype("Int64")

    subjects["subject_id"] = pd.to_numeric(
        subjects["subject_id"], errors="coerce"
    ).astype("Int64")

    # Remove metadata records with invalid or repeated IDs
    subjects = subjects.dropna(
        subset=["subject_id"]
    ).drop_duplicates(subset=["subject_id"])

    # Join participant information and detect unmatched IDs
    data = data.merge(
        subjects,
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    unmatched_subjects = int(
        (data["_merge"] == "left_only").sum()
    )

    # Remove the temporary merge indicator
    data = data.drop(columns=["_merge"])

    # Save the cleaned dataset
    output_path = OUTPUT_DIR / "motionsense_initial_clean.csv"
    data.to_csv(output_path, index=False)

    # Save a summary of the cleaning process
    summary = {
        "raw_rows": int(raw_rows),
        "rows_after_duplicate_removal": int(rows_after_cleaning),
        "duplicates_removed": int(duplicates_removed),
        "missing_values_before_imputation": missing_before,
        "remaining_missing_sensor_values": remaining_missing,
        "unmatched_participant_rows": unmatched_subjects,
        "sensor_columns": SENSOR_COLUMNS
    }

    summary_path = OUTPUT_DIR / "cleaning_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print(f"Initial cleaning complete: {len(data)} rows.")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Unmatched participant rows: {unmatched_subjects}")


# Run the cleaning process when this file is executed directly
if __name__ == "__main__":
    clean_data()
