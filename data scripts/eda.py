
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "motionsense_output"

SENSOR_COLUMNS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y",
    "userAcceleration.z"
]


def generate_eda():
    input_path = OUTPUT_DIR / "motionsense_cleaned.csv"
    data = pd.read_csv(input_path)

    print("Generating EDA charts...")

    sns.set_theme(style="whitegrid")

    # Activity distribution
    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=data,
        x="activity",
        order=data["activity"].value_counts().index
    )
    plt.title("MotionSense Activity Class Distribution")
    plt.xlabel("Activity")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "activity_distribution.png", dpi=300
    )
    plt.close()

    # Sensor correlation
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        data[SENSOR_COLUMNS].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Between Sensor Features")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sensor_correlation_heatmap.png",
        dpi=300
    )
    plt.close()

    # Sensor distributions
    axes = data[SENSOR_COLUMNS[:6]].hist(
        figsize=(12, 8), bins=30
    )
    plt.gcf().suptitle(
        "Distributions of Selected Sensor Features", y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sensor_feature_distributions.png",
        dpi=300
    )
    plt.close("all")

    print("EDA charts saved.")


if __name__ == "__main__":
    generate_eda()
