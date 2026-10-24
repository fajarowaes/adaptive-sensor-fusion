import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "motionsense_output"

SENSOR_COLUMNS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y",
    "userAcceleration.z"
]


def analyze_outliers():
    input_path = OUTPUT_DIR / "motionsense_initial_clean.csv"

    if not input_path.exists():
        raise FileNotFoundError("Run clean_data.py first.")

    data = pd.read_csv(input_path)

    missing_columns = [
        col for col in SENSOR_COLUMNS if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing sensor columns: {missing_columns}")

    print("Analyzing sensor outliers...")

    summary = {}

    for col in SENSOR_COLUMNS:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr <= 0:
            summary[col] = {
                "outliers_detected": 0,
                "outliers_capped": 0
            }
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (data[col] < lower) | (data[col] > upper)
        count = int(mask.sum())

        data[col] = data[col].clip(lower=lower, upper=upper)

        summary[col] = {
            "outliers_detected": count,
            "outliers_capped": count,
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }

    output_path = OUTPUT_DIR / "motionsense_cleaned.csv"
    data.to_csv(output_path, index=False)

    summary_path = OUTPUT_DIR / "outlier_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("Outlier analysis complete.")


if __name__ == "__main__":
    analyze_outliers()