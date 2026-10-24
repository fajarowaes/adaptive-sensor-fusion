
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set the project and output folder paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "motionsense_output"

# List all 12 sensor features
SENSOR_COLUMNS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y",
    "userAcceleration.z"
]


def preprocess_features():
    # Load the cleaned dataset
    input_path = OUTPUT_DIR / "motionsense_cleaned.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run outlier_analysis.py before preprocessing."
        )

    data = pd.read_csv(input_path)

    # Check that the required columns exist
    required_columns = SENSOR_COLUMNS + ["activity"]
    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    print("Encoding labels and scaling sensor features...")

    # Convert activity names into numeric labels
    encoder = LabelEncoder()
    data["activity_encoded"] = encoder.fit_transform(
        data["activity"].astype(str)
    )

    # Save the activity-to-number mapping
    label_mapping = {
        str(label): int(code)
        for label, code in zip(
            encoder.classes_,
            encoder.transform(encoder.classes_)
        )
    }

    mapping_path = OUTPUT_DIR / "activity_label_mapping.json"
    mapping_path.write_text(
        json.dumps(label_mapping, indent=4),
        encoding="utf-8"
    )

    # Convert sensor values to numeric format
    data[SENSOR_COLUMNS] = data[SENSOR_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    # Stop if any sensor values are missing
    if data[SENSOR_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing sensor values found. Check the cleaning stage."
        )

    # Standardize all 12 sensor features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        data[SENSOR_COLUMNS]
    )

    # Add scaled features while keeping the original columns
    scaled_columns = [
        f"{col}_scaled" for col in SENSOR_COLUMNS
    ]
    data[scaled_columns] = scaled_features

    # Save the fitted scaler for future sensor readings
    scaler_path = OUTPUT_DIR / "sensor_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Save the preprocessed dataset
    output_path = OUTPUT_DIR / "motionsense_preprocessed.csv"
    data.to_csv(output_path, index=False)

    print("Feature preprocessing complete.")
    print(f"Dataset saved to: {output_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Sensor features scaled: {len(SENSOR_COLUMNS)}")
    print(f"Activities found: {len(label_mapping)}")


# Run preprocessing when this file is executed directly
if __name__ == "__main__":
    preprocess_features()
