
"""
src/evaluate.py

Member 3: Model Evaluation for Adaptive Sensor Fusion HAR.

Evaluates four existing checkpoints on the untouched subject-wise test set.
No model training is performed.

Outputs:
    results/model_comparison.csv
    results/classification_report.csv
    results/baseline_confusion_matrix.png
    results/adaptive_confusion_matrix.png
    results/roc_curve.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "har_evaluation")
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataloaders, DEFAULT_LABEL_MAPPING_PATH
from src.run_experiments import load_experiment_model
from src.sensor_degradation import (
    DEGRADATION_CONDITIONS,
    DEGRADATION_REGISTRY,
    apply_condition,
    get_seeded_generator,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128

MODEL_CONFIGS = [
    ("baseline", "trained_on_normal"),
    ("adaptive", "trained_on_normal"),
    ("baseline", "degradation_aware"),
    ("adaptive", "degradation_aware"),
]

ACTIVITY_NAMES = [
    "downstairs",
    "jogging",
    "sitting",
    "standing",
    "upstairs",
    "walking",
]

NUM_CLASSES = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("member3_evaluation")


# ---------------------------------------------------------
# Label mapping
# ---------------------------------------------------------

def load_activity_labels():
    """Load the verified activity-to-integer mapping."""

    with open(DEFAULT_LABEL_MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    labels = sorted(mapping, key=mapping.get)

    if len(labels) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} activity classes, found {len(labels)}."
        )

    if sorted(mapping.values()) != list(range(NUM_CLASSES)):
        raise ValueError("Activity labels must be encoded from 0 to 5.")

    return labels


# ---------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred, probabilities):
    """Calculate classification metrics and multiclass OvR AUC."""

    labels = np.arange(NUM_CLASSES)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred, labels=labels,
        average="macro", zero_division=0
    )

    precision_weighted = precision_score(
        y_true, y_pred, labels=labels,
        average="weighted", zero_division=0
    )

    recall_macro = recall_score(
        y_true, y_pred, labels=labels,
        average="macro", zero_division=0
    )

    recall_weighted = recall_score(
        y_true, y_pred, labels=labels,
        average="weighted", zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_pred, labels=labels,
        average="macro", zero_division=0
    )

    f1_weighted = f1_score(
        y_true, y_pred, labels=labels,
        average="weighted", zero_division=0
    )

    auc_macro = roc_auc_score(
        y_true,
        probabilities,
        labels=labels,
        multi_class="ovr",
        average="macro"
    )

    auc_weighted = roc_auc_score(
        y_true,
        probabilities,
        labels=labels,
        multi_class="ovr",
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_ovr_macro": auc_macro,
        "auc_ovr_weighted": auc_weighted,
        "n_windows": len(y_true),
    }


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, labels, output_path, title):
    """Save a confusion matrix as a high-resolution PNG."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Activity",
        ylabel="Actual Activity",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor"
    )

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black"
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved confusion matrix: %s", output_path)


def save_roc_curves(y_true, model_outputs, labels, output_path):
    """Save one-vs-rest ROC curves for the two normally trained models."""

    fig, ax = plt.subplots(figsize=(10, 8))

    for model_key, output in model_outputs.items():

        probabilities = output["probabilities"]
        model_name = output["model"].capitalize()

        for class_index, class_name in enumerate(labels):

            binary_targets = (
                y_true == class_index
            ).astype(int)

            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_targets,
                probabilities[:, class_index]
            )

            class_auc = auc(
                false_positive_rate,
                true_positive_rate
            )

            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model_name} - {class_name} (AUC={class_auc:.3f})"
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="One-vs-Rest ROC Curves: Normal Sensor Conditions"
    )

    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved ROC curves: %s", output_path)


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def run_evaluation(seed=SEED, batch_size=BATCH_SIZE):

    logger.info("=" * 65)
    logger.info("MEMBER 3: MODEL EVALUATION")
    logger.info("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Evaluation device: %s", device)

    activity_labels = load_activity_labels()

    logger.info("Activity label order: %s", activity_labels)

    # Load the existing subject-wise test split.
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        seed=seed,
        num_workers=0
    )

    logger.info("Test windows: %d", len(test_loader.dataset))

    # Load all four existing trained checkpoints.
    models = {}

    for model_name, training_mode in MODEL_CONFIGS:

        model_key = f"{model_name}_{training_mode}"

        logger.info("Loading checkpoint: %s", model_key)

        model = load_experiment_model(
            model_name=model_name,
            training_mode=training_mode,
            device=device
        )

        model.eval()
        models[model_key] = model

    # Store predictions and probabilities for each model/condition.
    all_outputs = {}

    # Evaluate every degradation condition.
    for condition in DEGRADATION_CONDITIONS:

        logger.info("-" * 65)
        logger.info("Evaluating condition: %s", condition)

        condition_sigma = DEGRADATION_REGISTRY[
            condition
        ].default_sigma

        # One persistent generator per condition.
        # The same degraded batch is shared by all four models.
        generator = get_seeded_generator(
            condition=condition,
            seed=seed,
            device="cpu"
        )

        condition_targets = []
        condition_predictions = {
            key: [] for key in models
        }
        condition_probabilities = {
            key: [] for key in models
        }

        for batch_x, batch_y in test_loader:

            # Degrade the batch once, before evaluating any model.
            degraded_x, masks = apply_condition(
                x=batch_x,
                condition=condition,
                sigma=condition_sigma,
                generator=generator,
                seed=seed,
                compute_reliability=False
            )

            degraded_x = degraded_x.to(device)

            mask_acc = masks["mask_acc"].to(device)
            mask_gyro = masks["mask_gyro"].to(device)

            targets = batch_y.numpy()
            condition_targets.extend(targets.tolist())

            with torch.no_grad():

                for model_key, model in models.items():

                    if model_key.startswith("adaptive"):

                        logits = model(
                            x=degraded_x,
                            mask_acc=mask_acc,
                            mask_gyro=mask_gyro,
                            reliability_acc=None,
                            reliability_gyro=None
                        )

                    else:
                        logits = model(degraded_x)

                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )

                    predictions = torch.argmax(
                        probabilities,
                        dim=1
                    )

                    condition_predictions[model_key].extend(
                        predictions.cpu().numpy().tolist()
                    )

                    condition_probabilities[model_key].extend(
                        probabilities.cpu().numpy().tolist()
                    )

        y_true = np.asarray(condition_targets, dtype=np.int64)

        # Calculate and store metrics for every model.
        for model_key in models:

            y_pred = np.asarray(
                condition_predictions[model_key],
                dtype=np.int64
            )

            y_prob = np.asarray(
                condition_probabilities[model_key],
                dtype=np.float64
            )

            metrics = calculate_metrics(
                y_true,
                y_pred,
                y_prob
            )

            model_name, training_mode = model_key.split("_", 1)

            all_outputs[(model_key, condition)] = {
                "model": model_name,
                "training_mode": training_mode,
                "y_true": y_true,
                "y_pred": y_pred,
                "probabilities": y_prob,
                "metrics": metrics
            }

            logger.info(
                "%s | Acc: %.4f | Macro F1: %.4f | "
                "Weighted F1: %.4f | Macro AUC: %.4f",
                model_key,
                metrics["accuracy"],
                metrics["f1_macro"],
                metrics["f1_weighted"],
                metrics["auc_ovr_macro"]
            )

    # -----------------------------------------------------
    # Save model comparison CSV
    # -----------------------------------------------------

    comparison_records = []

    for (model_key, condition), output in all_outputs.items():

        record = {
            "condition": condition,
            "model": output["model"],
            "training_mode": output["training_mode"],
        }

        record.update(output["metrics"])
        comparison_records.append(record)

    comparison_df = pd.DataFrame(comparison_records)

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    logger.info("Saved model comparison: %s", comparison_path)

    # -----------------------------------------------------
    # Save classification report for normal conditions
    # -----------------------------------------------------

    report_records = []

    for model_name, training_mode in MODEL_CONFIGS:

        model_key = f"{model_name}_{training_mode}"

        output = all_outputs[(model_key, "normal")]

        report = classification_report(
            output["y_true"],
            output["y_pred"],
            labels=np.arange(NUM_CLASSES),
            target_names=activity_labels,
            output_dict=True,
            zero_division=0
        )

        for class_name, values in report.items():

            if isinstance(values, dict):

                report_records.append({
                    "model": model_name,
                    "training_mode": training_mode,
                    "activity": class_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1-score"],
                    "support": values["support"]
                })

            else:

                report_records.append({
                    "model": model_name,
                    "training_mode": training_mode,
                    "activity": class_name,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": values,
                    "support": len(output["y_true"])
                })

    report_df = pd.DataFrame(report_records)

    report_path = RESULTS_DIR / "classification_report.csv"

    report_df.to_csv(
        report_path,
        index=False
    )

    logger.info("Saved classification report: %s", report_path)

    # -----------------------------------------------------
    # Save individual predictions with participant metadata
    # -----------------------------------------------------

    prediction_records = []

    # Metadata order matches the test DataLoader because shuffle=False.
    test_metadata = test_loader.dataset.metadata

    if len(test_metadata) != len(test_loader.dataset):
        raise ValueError("Test metadata and test dataset lengths do not match.")

    for (model_key, condition), output in all_outputs.items():

        model_name = output["model"]
        training_mode = output["training_mode"]

        y_true = output["y_true"]
        y_pred = output["y_pred"]
        probabilities = output["probabilities"]

        condition_sigma = DEGRADATION_REGISTRY[
            condition
        ].default_sigma

        for index, metadata in enumerate(test_metadata):

            true_label = int(y_true[index])
            predicted_label = int(y_pred[index])

            record = {
                "model": model_name,
                "training_mode": training_mode,
                "condition": condition,
                "noise_sigma": condition_sigma,
                "subject_id": metadata["subject_id"],
                "trial": metadata["trial"],
                "source_file": metadata["source_file"],
                "start_sample_index": metadata["start_sample_index"],
                "end_sample_index": metadata["end_sample_index"],
                "true_label": true_label,
                "true_activity": activity_labels[true_label],
                "predicted_label": predicted_label,
                "predicted_activity": activity_labels[predicted_label],
                "correct": int(true_label == predicted_label),
            }

            for class_index, class_name in enumerate(activity_labels):
                record[f"prob_{class_name}"] = float(
                    probabilities[index, class_index]
                )

            prediction_records.append(record)

    predictions_df = pd.DataFrame(prediction_records)

    predictions_path = RESULTS_DIR / "predictions.csv"

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    logger.info(
        "Saved individual predictions: %s",
        predictions_path
    )


    # -----------------------------------------------------
    # Save confusion matrices for normally trained models
    # -----------------------------------------------------

    baseline_output = all_outputs[
        ("baseline_trained_on_normal", "normal")
    ]

    adaptive_output = all_outputs[
        ("adaptive_trained_on_normal", "normal")
    ]

    save_confusion_matrix(
        baseline_output["y_true"],
        baseline_output["y_pred"],
        activity_labels,
        RESULTS_DIR / "baseline_confusion_matrix.png",
        "Baseline CNN: Normal Sensor Conditions"
    )

    save_confusion_matrix(
        adaptive_output["y_true"],
        adaptive_output["y_pred"],
        activity_labels,
        RESULTS_DIR / "adaptive_confusion_matrix.png",
        "Adaptive Sensor Fusion CNN: Normal Sensor Conditions"
    )

    # -----------------------------------------------------
    # Save ROC curves for normally trained models
    # -----------------------------------------------------

    normal_roc_outputs = {
        "baseline": baseline_output,
        "adaptive": adaptive_output
    }

    save_roc_curves(
        baseline_output["y_true"],
        normal_roc_outputs,
        activity_labels,
        RESULTS_DIR / "roc_curve.png"
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    logger.info("=" * 65)
    logger.info("EVALUATION COMPLETED")
    logger.info("Test windows evaluated: %d", len(test_loader.dataset))
    logger.info("Models evaluated: %d", len(models))
    logger.info("Degradation conditions: %d", len(DEGRADATION_CONDITIONS))
    logger.info("Comparison CSV: %s", comparison_path)
    logger.info("Classification report: %s", report_path)
    logger.info("=" * 65)

    print("\nNORMAL-CONDITION TEST RESULTS")
    print(
        comparison_df[
            (comparison_df["condition"] == "normal")
        ][
            [
                "model",
                "training_mode",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auc_ovr_macro"
            ]
        ].to_string(index=False)
    )

    return comparison_df, report_df, all_outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate existing HAR model checkpoints."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )

    args = parser.parse_args()

    run_evaluation(
        seed=args.seed,
        batch_size=args.batch_size
    )
