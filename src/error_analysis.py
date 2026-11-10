
"""
Member 3: Error Analysis
Analyzes saved test predictions across all models and sensor conditions.
Does not load models or perform retraining.
"""

from pathlib import Path
import logging

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"

ACTIVITY_LABELS = [
    "downstairs",
    "jogging",
    "sitting",
    "standing",
    "upstairs",
    "walking",
]

LABEL_IDS = list(range(len(ACTIVITY_LABELS)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("member3_error_analysis")


# ---------------------------------------------------------
# Load and validate predictions
# ---------------------------------------------------------

def load_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}\n"
            "Run python -u src\\evaluate.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "model",
        "training_mode",
        "condition",
        "true_label",
        "predicted_label",
        "correct",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["true_label"] = df["true_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)

    if not df["true_label"].isin(LABEL_IDS).all():
        raise ValueError("Unexpected true activity labels found.")

    if not df["predicted_label"].isin(LABEL_IDS).all():
        raise ValueError("Unexpected predicted activity labels found.")

    logger.info("Loaded %s prediction records.", len(df))

    return df


# ---------------------------------------------------------
# Class-wise error analysis
# ---------------------------------------------------------

def calculate_class_errors(df):

    records = []

    group_columns = [
        "model",
        "training_mode",
        "condition",
    ]

    grouped = df.groupby(group_columns, sort=True)

    for (model, training_mode, condition), group in grouped:

        y_true = group["true_label"].to_numpy()
        y_pred = group["predicted_label"].to_numpy()

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=LABEL_IDS,
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=LABEL_IDS,
            target_names=ACTIVITY_LABELS,
            output_dict=True,
            zero_division=0,
        )

        for class_id, activity in enumerate(ACTIVITY_LABELS):

            true_positive = int(matrix[class_id, class_id])

            false_negative = int(
                matrix[class_id, :].sum() - true_positive
            )

            false_positive = int(
                matrix[:, class_id].sum() - true_positive
            )

            class_metrics = report[activity]

            records.append({
                "model": model,
                "training_mode": training_mode,
                "condition": condition,
                "activity": activity,
                "support": int(class_metrics["support"]),
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1-score"],
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Identify most confused activity pairs
# ---------------------------------------------------------

def calculate_confused_pairs(df):

    records = []

    group_columns = [
        "model",
        "training_mode",
        "condition",
    ]

    grouped = df.groupby(group_columns, sort=True)

    for (model, training_mode, condition), group in grouped:

        matrix = confusion_matrix(
            group["true_label"],
            group["predicted_label"],
            labels=LABEL_IDS,
        )

        for true_id in LABEL_IDS:

            for predicted_id in LABEL_IDS:

                if true_id == predicted_id:
                    continue

                count = int(matrix[true_id, predicted_id])

                if count == 0:
                    continue

                records.append({
                    "model": model,
                    "training_mode": training_mode,
                    "condition": condition,
                    "actual_activity": ACTIVITY_LABELS[true_id],
                    "predicted_activity": ACTIVITY_LABELS[predicted_id],
                    "misclassified_windows": count,
                })

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            by=[
                "model",
                "training_mode",
                "condition",
                "misclassified_windows",
            ],
            ascending=[True, True, True, False],
        )

        result["confusion_rank"] = (
            result.groupby(
                ["model", "training_mode", "condition"]
            )["misclassified_windows"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        result = result[
            [
                "model",
                "training_mode",
                "condition",
                "confusion_rank",
                "actual_activity",
                "predicted_activity",
                "misclassified_windows",
            ]
        ]

    return result


# ---------------------------------------------------------
# Overall performance and degradation error summary
# ---------------------------------------------------------

def calculate_degradation_summary(df):

    records = []

    group_columns = [
        "model",
        "training_mode",
        "condition",
    ]

    for (model, training_mode, condition), group in df.groupby(
        group_columns,
        sort=True,
    ):

        y_true = group["true_label"].to_numpy()
        y_pred = group["predicted_label"].to_numpy()

        accuracy = accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=LABEL_IDS,
            average="macro",
            zero_division=0,
        )

        records.append({
            "model": model,
            "training_mode": training_mode,
            "condition": condition,
            "test_windows": len(group),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "error_rate": 1 - accuracy,
        })

    summary = pd.DataFrame(records)

    normal_errors = (
        summary[summary["condition"] == "normal"][
            ["model", "training_mode", "error_rate"]
        ]
        .rename(columns={
            "error_rate": "normal_error_rate"
        })
    )

    summary = summary.merge(
        normal_errors,
        on=["model", "training_mode"],
        how="left",
    )

    summary["error_rate_change"] = (
        summary["error_rate"] - summary["normal_error_rate"]
    )

    summary["error_rate_change_percentage_points"] = (
        summary["error_rate_change"] * 100
    )

    return summary


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def run_error_analysis():

    logger.info("=" * 65)
    logger.info("MEMBER 3: ERROR ANALYSIS")
    logger.info("=" * 65)

    df = load_predictions()

    # Class-wise errors
    class_errors = calculate_class_errors(df)

    class_errors_path = RESULTS_DIR / "error_analysis.csv"

    class_errors.to_csv(
        class_errors_path,
        index=False,
    )

    logger.info("Saved class-wise analysis: %s", class_errors_path)

    # Most confused activity pairs
    confused_pairs = calculate_confused_pairs(df)

    confused_pairs_path = (
        RESULTS_DIR / "confused_activity_pairs.csv"
    )

    confused_pairs.to_csv(
        confused_pairs_path,
        index=False,
    )

    logger.info("Saved confusion pairs: %s", confused_pairs_path)

    # Overall degradation error summary
    degradation_summary = calculate_degradation_summary(df)

    degradation_path = (
        RESULTS_DIR / "degradation_error_summary.csv"
    )

    degradation_summary.to_csv(
        degradation_path,
        index=False,
    )

    logger.info(
        "Saved degradation error summary: %s",
        degradation_path,
    )

    # Display normal-condition class-wise results
    print("\nNORMAL-CONDITION CLASS-WISE ERROR ANALYSIS")

    normal_results = class_errors[
        class_errors["condition"] == "normal"
    ]

    print(
        normal_results[
            [
                "model",
                "training_mode",
                "activity",
                "precision",
                "recall",
                "f1_score",
                "false_positive",
                "false_negative",
            ]
        ].to_string(index=False)
    )

    # Display top three confusion pairs per configuration
    print("\nTOP THREE CONFUSIONS PER MODEL AND CONDITION")

    if not confused_pairs.empty:

        top_confusions = confused_pairs[
            confused_pairs["confusion_rank"] <= 3
        ]

        print(top_confusions.to_string(index=False))

    # Display degradation summary
    print("\nDEGRADATION ERROR SUMMARY")

    print(
        degradation_summary[
            [
                "model",
                "training_mode",
                "condition",
                "accuracy",
                "macro_f1",
                "error_rate",
                "error_rate_change_percentage_points",
            ]
        ].to_string(index=False)
    )

    logger.info("=" * 65)
    logger.info("ERROR ANALYSIS COMPLETED")
    logger.info("Prediction records analyzed: %s", len(df))
    logger.info("Class-wise records: %s", len(class_errors))
    logger.info("Confusion pairs: %s", len(confused_pairs))
    logger.info("Performance summaries: %s", len(degradation_summary))
    logger.info("=" * 65)


if __name__ == "__main__":
    run_error_analysis()
