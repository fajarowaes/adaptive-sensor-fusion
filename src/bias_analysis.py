
"""
Member 3: Demographic Bias Analysis

Analyzes model performance across participant age and gender groups.
Uses normal-condition test predictions only.

Important:
- Gender codes are kept numeric because their meanings have not
  been verified from authoritative dataset documentation.
- Only four participants are present in the test set.
- Results are exploratory, not conclusive fairness claims.
"""

from pathlib import Path
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
DEMOGRAPHICS_PATH = (
    PROJECT_ROOT
    / "motionsense_output"
    / "data_subjects_info.csv"
)

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

logger = logging.getLogger("member3_bias_analysis")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    if not DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(
            f"Demographics file not found: {DEMOGRAPHICS_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)

    demographics = pd.read_csv(DEMOGRAPHICS_PATH)

    required_prediction_columns = [
        "model",
        "training_mode",
        "condition",
        "subject_id",
        "true_label",
        "predicted_label",
    ]

    required_demographic_columns = [
        "code",
        "age",
        "gender",
    ]

    for column in required_prediction_columns:
        if column not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {column}"
            )

    for column in required_demographic_columns:
        if column not in demographics.columns:
            raise ValueError(
                f"Missing demographic column: {column}"
            )

    predictions["subject_id"] = (
        predictions["subject_id"].astype(int)
    )

    demographics["code"] = demographics["code"].astype(int)

    predictions["true_label"] = (
        predictions["true_label"].astype(int)
    )

    predictions["predicted_label"] = (
        predictions["predicted_label"].astype(int)
    )

    demographics["age"] = pd.to_numeric(
        demographics["age"],
        errors="raise",
    )

    demographics["gender"] = pd.to_numeric(
        demographics["gender"],
        errors="raise",
    )

    # Analyze clean, normal-condition predictions only.
    predictions = predictions[
        predictions["condition"] == "normal"
    ].copy()

    merged = predictions.merge(
        demographics[
            ["code", "age", "gender"]
        ],
        left_on="subject_id",
        right_on="code",
        how="left",
        validate="many_to_one",
    )

    if merged["age"].isna().any() or merged["gender"].isna().any():
        raise ValueError(
            "Some test participants do not have demographic metadata."
        )

    merged["age_band"] = merged["age"].apply(
        lambda age: "18-25" if age <= 25 else "26+"
    )

    logger.info(
        "Loaded %s normal-condition prediction records.",
        len(merged),
    )

    logger.info(
        "Test participants: %s",
        sorted(merged["subject_id"].unique().tolist()),
    )

    return merged


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(group):

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

    return accuracy, macro_f1


# ---------------------------------------------------------
# Analyze participant and demographic groups
# ---------------------------------------------------------

def analyze_group(
    group,
    model,
    training_mode,
    group_type,
    group_label,
):

    participant_metrics = []

    for subject_id, participant_data in group.groupby(
        "subject_id"
    ):

        accuracy, macro_f1 = calculate_metrics(
            participant_data
        )

        participant_metrics.append({
            "subject_id": int(subject_id),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "windows": len(participant_data),
        })

    participant_df = pd.DataFrame(participant_metrics)

    pooled_accuracy, pooled_macro_f1 = calculate_metrics(
        group
    )

    return {
        "model": model,
        "training_mode": training_mode,
        "condition": "normal",
        "group_type": group_type,
        "group": group_label,
        "participant_count": group["subject_id"].nunique(),
        "participant_ids": ", ".join(
            map(
                str,
                sorted(group["subject_id"].unique()),
            )
        ),
        "window_count": len(group),
        "pooled_accuracy": pooled_accuracy,
        "pooled_macro_f1": pooled_macro_f1,
        "mean_participant_accuracy": (
            participant_df["accuracy"].mean()
        ),
        "mean_participant_macro_f1": (
            participant_df["macro_f1"].mean()
        ),
        "std_participant_accuracy": (
            participant_df["accuracy"].std(ddof=1)
            if len(participant_df) > 1
            else 0.0
        ),
        "min_participant_accuracy": (
            participant_df["accuracy"].min()
        ),
        "max_participant_accuracy": (
            participant_df["accuracy"].max()
        ),
    }


# ---------------------------------------------------------
# Build demographic analysis
# ---------------------------------------------------------

def run_bias_analysis():

    logger.info("=" * 65)
    logger.info("MEMBER 3: DEMOGRAPHIC BIAS ANALYSIS")
    logger.info("=" * 65)

    df = load_data()

    records = []

    model_groups = df.groupby(
        ["model", "training_mode"],
        sort=True,
    )

    for (model, training_mode), model_data in model_groups:

        # -------------------------------------------------
        # Individual participant performance
        # -------------------------------------------------

        for subject_id, participant_data in model_data.groupby(
            "subject_id"
        ):

            records.append(
                analyze_group(
                    participant_data,
                    model,
                    training_mode,
                    "participant",
                    f"subject_{subject_id}",
                )
            )

        # -------------------------------------------------
        # Gender group performance
        # -------------------------------------------------

        for gender_code, gender_data in model_data.groupby(
            "gender"
        ):

            records.append(
                analyze_group(
                    gender_data,
                    model,
                    training_mode,
                    "gender",
                    f"gender_{int(gender_code)}",
                )
            )

        # -------------------------------------------------
        # Age group performance
        # -------------------------------------------------

        for age_band, age_data in model_data.groupby(
            "age_band"
        ):

            records.append(
                analyze_group(
                    age_data,
                    model,
                    training_mode,
                    "age_band",
                    f"age_{age_band}",
                )
            )

    result = pd.DataFrame(records)

    output_path = RESULTS_DIR / "bias_analysis.csv"

    result.to_csv(
        output_path,
        index=False,
    )

    logger.info("Saved demographic analysis: %s", output_path)

    # -----------------------------------------------------
    # Display subgroup results
    # -----------------------------------------------------

    print("\nDEMOGRAPHIC GROUP PERFORMANCE")

    subgroup_results = result[
        result["group_type"].isin(
            ["gender", "age_band"]
        )
    ]

    print(
        subgroup_results[
            [
                "model",
                "training_mode",
                "group_type",
                "group",
                "participant_count",
                "window_count",
                "pooled_accuracy",
                "pooled_macro_f1",
                "mean_participant_accuracy",
                "mean_participant_macro_f1",
            ]
        ].to_string(index=False)
    )

    print("\nINDIVIDUAL PARTICIPANT PERFORMANCE")

    participant_results = result[
        result["group_type"] == "participant"
    ]

    print(
        participant_results[
            [
                "model",
                "training_mode",
                "group",
                "window_count",
                "pooled_accuracy",
                "pooled_macro_f1",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Sample-size warning
    # -----------------------------------------------------

    print("\nBIAS ANALYSIS LIMITATIONS")
    print(
        f"Number of test participants: "
        f"{df['subject_id'].nunique()}"
    )

    print(
        "Gender group participant counts:"
    )

    print(
        df[["subject_id", "gender"]]
        .drop_duplicates()
        .groupby("gender")["subject_id"]
        .nunique()
        .to_string()
    )

    print(
        "Age group participant counts:"
    )

    print(
        df[["subject_id", "age_band"]]
        .drop_duplicates()
        .groupby("age_band")["subject_id"]
        .nunique()
        .to_string()
    )

    logger.info("=" * 65)
    logger.info("BIAS ANALYSIS COMPLETED")
    logger.info("Total analysis records: %s", len(result))
    logger.info("=" * 65)


if __name__ == "__main__":
    run_bias_analysis()
