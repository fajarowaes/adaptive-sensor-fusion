
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

# URLs of the MotionSense sensor data and participant information
DATA_URL = (
    "https://github.com/mmalekzadeh/motion-sense/"
    "raw/master/data/A_DeviceMotion_data.zip"
)

SUBJECT_URL = (
    "https://raw.githubusercontent.com/mmalekzadeh/"
    "motion-sense/master/data/data_subjects_info.csv"
)

# Set the project and output folder paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "motionsense_output"

# Required columns in the participant information file
EXPECTED_SUBJECT_COLUMNS = {
    "code", "weight", "height", "age", "gender"
}


def download_file(url, destination, timeout=120):
    # Download to a temporary file to avoid saving incomplete files
    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    try:
        # Download the file in chunks to reduce memory usage
        with requests.get(
            url, stream=True, timeout=timeout
        ) as response:
            response.raise_for_status()

            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        # Check that the downloaded file is not empty
        if temporary_path.stat().st_size == 0:
            raise ValueError(
                f"Downloaded file is empty: {destination.name}"
            )

        # Move the completed download to its final location
        temporary_path.replace(destination)

    finally:
        # Remove any temporary file left after an error
        if temporary_path.exists():
            temporary_path.unlink()


def validate_sensor_zip(zip_path):
    # Check whether the downloaded file is a valid ZIP archive
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Downloaded sensor file is not a valid ZIP.")

    with zipfile.ZipFile(zip_path, "r") as archive:
        # Find sensor CSV files inside the dataset folder
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and "A_DeviceMotion_data/" in name
            and not name.endswith("/")
        ]

        if not csv_files:
            raise ValueError(
                "No sensor CSV files found in A_DeviceMotion_data."
            )

        # List the sensor features expected in the dataset
        required_features = {
            "attitude.roll", "attitude.pitch", "attitude.yaw",
            "gravity.x", "gravity.y", "gravity.z",
            "rotationRate.x", "rotationRate.y", "rotationRate.z",
            "userAcceleration.x", "userAcceleration.y",
            "userAcceleration.z"
        }

        # Read the column names from one sensor CSV file
        with archive.open(csv_files[0]) as file:
            header = next(csv.reader(
                file.read().decode("utf-8-sig").splitlines()
            ))

        # Check that all required sensor features are present
        if not required_features.issubset(set(header)):
            raise ValueError(
                "Sensor CSV is missing expected feature columns."
            )

        # Return the total number of sensor CSV files
        return len(csv_files)


def validate_subjects(subject_path):
    # Read the participant information into a DataFrame
    subjects = pd.read_csv(subject_path)

    # Find any required columns missing from the file
    missing_columns = (
        EXPECTED_SUBJECT_COLUMNS - set(subjects.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Participant CSV is missing columns: {missing_columns}"
        )

    # Make sure the file contains participant records
    if subjects.empty:
        raise ValueError("Participant CSV contains no records.")

    return len(subjects)


def download_data():
    # Create the output folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set the paths for the downloaded files
    sensor_path = OUTPUT_DIR / "motionsense_raw.zip"
    subjects_path = OUTPUT_DIR / "data_subjects_info.csv"

    # Download and validate the sensor dataset
    print("Downloading MotionSense sensor dataset...")
    download_file(DATA_URL, sensor_path)

    print("Validating sensor ZIP...")
    sensor_csv_count = validate_sensor_zip(sensor_path)

    # Download and validate participant information
    print("Downloading participant information...")
    download_file(SUBJECT_URL, subjects_path, timeout=60)

    print("Validating participant information...")
    participant_count = validate_subjects(subjects_path)

    # Store download details and validation results in a report
    report = {
        "dataset": "MotionSense",
        "sensor_data_url": DATA_URL,
        "participant_data_url": SUBJECT_URL,
        "download_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sensor_zip_size_bytes": sensor_path.stat().st_size,
        "sensor_csv_files_found": sensor_csv_count,
        "participant_records_downloaded": participant_count,
        "validation_status": "passed"
    }

    # Save the report as a JSON file
    report_path = OUTPUT_DIR / "collection_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("Download and validation complete.")
    print(f"Sensor CSV files found: {sensor_csv_count}")
    print(f"Participants found: {participant_count}")
    print(f"Report saved to: {report_path}")


# Run the download process when this file is executed directly
if __name__ == "__main__":
    download_data()
