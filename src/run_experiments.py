#!.venv/Scripts/python.exe
"""
src/run_experiments.py

Phase 3/4 Orchestration Pipeline for Human Activity Recognition (MotionSense).
Owned by Member 2.

Order of Execution:
    1. FINAL TRAINING:
       Reads results/best_hyperparameters.json and trains BaselineCNN and
       AdaptiveSensorFusion under their selected hyperparameters with early stopping:
       - trained_on_normal -> models/baseline_best.pth, models/adaptive_best.pth
       - degradation_aware -> models/baseline_degradation_aware.pth, models/adaptive_degradation_aware.pth
       Generates results/loss_curve.png and results/accuracy_curve.png.
    2. DEGRADATION EVALUATION:
       Evaluates all models on VALIDATION and TEST sets under all 6 conditions:
       (normal, noisy_acc, noisy_gyro, missing_acc, missing_gyro, noisy_both).
       Computes Accuracy, Macro Precision, Macro Recall, Macro F1.
       Writes results/degradation_results.csv.
       Collects and saves adaptive fusion weights to results/adaptive_fusion_weights.csv.
    3. SIGMA SWEEP:
       Sweeps noise standard deviation sigma in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
       for noisy_acc, noisy_gyro, and noisy_both on the TEST set.
       Writes results/noise_sweep.csv.
    4. PUBLICATION-QUALITY PLOTS (300 DPI):
       - results/degradation_accuracy_comparison.png
       - results/degradation_f1_comparison.png
       - results/noise_sweep.png
       - results/confusion_matrix_normal_baseline.png
       - results/confusion_matrix_normal_adaptive.png
       - results/adaptive_fusion_weights.png
    5. FACTUAL SUMMARY:
       Prints an objective summary table comparing performance without bias.

Hard Rules:
    - Never invent or hardcode results; all numbers come from executed code.
    - Test set is touched ONLY during final evaluation, never for tuning or early stopping.
    - Fixed seed (42) for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Matplotlib headless config
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_har_exp"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.data_loader import (
    DEFAULT_LABEL_MAPPING_PATH,
    get_dataloaders,
)
from src.model import BaselineCNN, load_model
from src.sensor_degradation import (
    DEGRADATION_CONDITIONS,
    DEGRADATION_REGISTRY,
    SIGMA_SWEEP_VALUES,
    apply_condition,
    forward_with_condition,
    get_seeded_generator,
    is_adaptive_fusion_model,
)
from src.sensor_fusion import AdaptiveSensorFusion, load_adaptive_model
from src.train import build_model, plot_curves, train_model

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiments")

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_HP_PATH = RESULTS_DIR / "best_hyperparameters.json"


# ===========================================================================
# 1. Final Training Routine
# ===========================================================================

def load_best_hyperparameters() -> Dict[str, Any]:
    """Loads selected hyperparameters from results/best_hyperparameters.json."""
    if not BEST_HP_PATH.exists():
        raise FileNotFoundError(
            f"Hyperparameter configuration file not found at: {BEST_HP_PATH}. "
            f"Please run src/hyperparameter_tuning.py first."
        )
    with open(BEST_HP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded best hyperparameters from: {BEST_HP_PATH}")
    return data


def run_final_training(
    models: List[str],
    train_modes: List[str],
    seeds: List[int],
    smoke: bool = False,
    smoke_epochs: int = 2,
    smoke_samples: int = 256,
) -> Dict[str, Any]:
    """
    Executes final training runs using tuned hyperparameters.

    Produces:
        - models/baseline_best.pth & models/baseline_config.json
        - models/adaptive_best.pth & models/adaptive_config.json
        - models/baseline_degradation_aware.pth & models/baseline_degradation_aware_config.json
        - models/adaptive_degradation_aware.pth & models/adaptive_degradation_aware_config.json
        - results/training_history.csv
        - results/loss_curve.png & results/accuracy_curve.png
    """
    best_hp = load_best_hyperparameters()
    training_summaries: List[Dict[str, Any]] = []

    for seed in seeds:
        for model_name in models:
            if model_name not in best_hp:
                logger.warning(f"No hyperparameter config found for '{model_name}'. Skipping.")
                continue

            hp = best_hp[model_name]
            lr = float(hp["lr"])
            batch_size = int(hp["batch_size"])
            dropout = float(hp["dropout"])
            optimizer_name = str(hp["optimizer"])
            max_epochs = smoke_epochs if smoke else int(hp["max_epochs"])

            for mode in train_modes:
                # mode: 'trained_on_normal' (clean training) or 'degradation_aware'
                is_deg_aware = (mode == "degradation_aware")
                train_cond = "degradation_aware" if is_deg_aware else "normal"

                ckpt_name = f"{model_name}_degradation_aware.pth" if is_deg_aware else f"{model_name}_best.pth"
                cfg_name = f"{model_name}_degradation_aware_config.json" if is_deg_aware else f"{model_name}_config.json"

                logger.info(
                    f"\n{'='*70}\n"
                    f"  TRAINING: {model_name.upper()} | Mode: {mode} | Seed: {seed}\n"
                    f"  lr={lr}, bs={batch_size}, do={dropout}, opt={optimizer_name}, max_ep={max_epochs}\n"
                    f"{'='*70}"
                )

                t0 = time.time()
                res = train_model(
                    model_name=model_name,
                    train_condition=train_cond,
                    lr=lr,
                    batch_size=batch_size,
                    dropout=dropout,
                    max_epochs=max_epochs,
                    optimizer_name=optimizer_name,
                    seed=seed,
                    patience=7,
                    use_scheduler=False,
                    grad_clip=None,
                    output_tag=f"final_{mode}_seed{seed}",
                    results_dir=RESULTS_DIR,
                    models_dir=MODELS_DIR,
                    smoke_test=smoke,
                    smoke_samples=smoke_samples,
                    checkpoint_filename=ckpt_name,
                    config_filename=cfg_name,
                )
                elapsed = time.time() - t0

                training_summaries.append({
                    "model": model_name,
                    "training_mode": mode,
                    "seed": seed,
                    "best_epoch": res.get("best_epoch"),
                    "best_val_acc": res.get("best_val_acc"),
                    "best_val_loss": res.get("best_val_loss"),
                    "elapsed_s": round(elapsed, 1),
                })

    # Generate training curves
    try:
        hist_csv = RESULTS_DIR / "training_history.csv"
        if hist_csv.exists():
            loss_p, acc_p = plot_curves(history_csv=hist_csv, out_dir=RESULTS_DIR)
            logger.info(f"Generated training curves:\n  - {loss_p}\n  - {acc_p}")
    except Exception as exc:
        logger.warning(f"Could not generate training curves: {exc}")

    return {"training_summaries": training_summaries}


# ===========================================================================
# 2. Model Loader Utility
# ===========================================================================

def load_experiment_model(
    model_name: str,
    training_mode: str,
    device: torch.device,
) -> nn.Module:
    """
    Instantiates and loads weights for a model variant.

    Args:
        model_name: 'baseline' or 'adaptive'
        training_mode: 'trained_on_normal' or 'degradation_aware'
        device: torch device

    Returns:
        Loaded model in eval mode.
    """
    best_hp = load_best_hyperparameters()
    hp = best_hp.get(model_name, {})
    dropout = float(hp.get("dropout", 0.3))

    model = build_model(model_name, dropout=dropout).to(device)

    is_deg_aware = (training_mode == "degradation_aware")
    ckpt_name = f"{model_name}_degradation_aware.pth" if is_deg_aware else f"{model_name}_best.pth"
    ckpt_path = MODELS_DIR / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {ckpt_path}. Run training first."
        )

    if model_name == "adaptive":
        load_adaptive_model(model, str(ckpt_path), map_location=str(device))
    else:
        load_model(model, str(ckpt_path), map_location=str(device))

    model.eval()
    logger.info(f"Loaded {model_name} ({training_mode}) from {ckpt_path.name}")
    return model


# ===========================================================================
# 3. Degradation Evaluation (6 Conditions on Val and Test)
# ===========================================================================

def evaluate_batch(
    model: nn.Module,
    x: torch.Tensor,
    condition: str,
    sigma: Optional[float],
    seed: int,
    device: torch.device,
    return_weights: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Runs forward pass for a single batch under degradation.

    Returns:
        (predictions_array, weights_dict_or_None)
    """
    x = x.to(device)
    is_adaptive = is_adaptive_fusion_model(model)

    with torch.no_grad():
        if return_weights and is_adaptive:
            preds, w_info = forward_with_condition(
                model=model,
                x=x,
                condition=condition,
                sigma=sigma,
                seed=seed,
                return_weights=True,
                mode="predict",
            )
            # w_info has 'w_acc', 'w_gyro' of shape (B, 1)
            w_acc = w_info["w_acc"].detach().cpu().numpy().flatten()
            w_gyro = w_info["w_gyro"].detach().cpu().numpy().flatten()
            return preds.detach().cpu().numpy(), {"w_acc": w_acc, "w_gyro": w_gyro}
        else:
            preds = forward_with_condition(
                model=model,
                x=x,
                condition=condition,
                sigma=sigma,
                seed=seed,
                return_weights=False,
                mode="predict",
            )
            return preds.detach().cpu().numpy(), None


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    condition: str,
    sigma: Optional[float],
    seed: int,
    device: torch.device,
    collect_weights: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Evaluates a model over an entire DataLoader under specified degradation.

    Returns:
        metrics_dict, y_true_all, y_pred_all, weights_summary_dict
    """
    all_preds = []
    all_targets = []
    w_acc_list = []
    w_gyro_list = []

    for batch_x, batch_y in loader:
        targets = batch_y.numpy()
        preds, w_data = evaluate_batch(
            model=model,
            x=batch_x,
            condition=condition,
            sigma=sigma,
            seed=seed,
            device=device,
            return_weights=collect_weights,
        )
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

        if w_data is not None:
            w_acc_list.extend(w_data["w_acc"].tolist())
            w_gyro_list.extend(w_data["w_gyro"].tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": round(acc, 5),
        "precision_macro": round(prec, 5),
        "recall_macro": round(rec, 5),
        "f1_macro": round(f1, 5),
        "n_windows": len(y_true),
    }

    w_summary = None
    if w_acc_list and w_gyro_list:
        w_summary = {
            "w_acc_mean": float(np.mean(w_acc_list)),
            "w_acc_std": float(np.std(w_acc_list)),
            "w_gyro_mean": float(np.mean(w_gyro_list)),
            "w_gyro_std": float(np.std(w_gyro_list)),
        }

    return metrics, y_true, y_pred, w_summary


def run_degradation_evaluation(
    models: List[str],
    train_modes: List[str],
    seed: int = 42,
    batch_size: int = 128,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates all model variants across 6 degradation conditions on VAL and TEST splits.

    Outputs:
        - results/degradation_results.csv
        - results/adaptive_fusion_weights.csv
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Degradation Evaluation on device: {device}")

    # Load dataloaders
    _, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )

    splits = {"val": val_loader, "test": test_loader}

    results_records: List[Dict[str, Any]] = []
    adaptive_weights_records: List[Dict[str, Any]] = []
    predictions_cache: Dict[str, Any] = {}

    for model_name in models:
        for t_mode in train_modes:
            logger.info(f"\n--- Evaluating {model_name.upper()} ({t_mode}) ---")
            try:
                model = load_experiment_model(model_name, t_mode, device)
            except FileNotFoundError as err:
                logger.error(f"Cannot evaluate {model_name} ({t_mode}): {err}")
                continue

            for split_name, loader in splits.items():
                for cond in DEGRADATION_CONDITIONS:
                    spec = DEGRADATION_REGISTRY[cond]
                    sigma = spec.default_sigma
                    collect_weights = (model_name == "adaptive" and split_name == "test")

                    metrics, y_true, y_pred, w_summary = evaluate_dataset(
                        model=model,
                        loader=loader,
                        condition=cond,
                        sigma=sigma,
                        seed=seed,
                        device=device,
                        collect_weights=collect_weights,
                    )

                    rec = {
                        "model": model_name,
                        "training_mode": t_mode,
                        "condition": cond,
                        "split": split_name,
                        "sigma": sigma,
                        "accuracy": metrics["accuracy"],
                        "precision_macro": metrics["precision_macro"],
                        "recall_macro": metrics["recall_macro"],
                        "f1_macro": metrics["f1_macro"],
                        "n_windows": metrics["n_windows"],
                    }
                    results_records.append(rec)

                    logger.info(
                        f"[{split_name.upper()}] {model_name} ({t_mode}) | {cond:12s} "
                        f"(sigma={sigma:.2f}) -> Acc: {metrics['accuracy']*100:.2f}%, "
                        f"F1: {metrics['f1_macro']:.4f}"
                    )

                    # Save weights for adaptive model on test set
                    if w_summary is not None:
                        adaptive_weights_records.append({
                            "training_mode": t_mode,
                            "condition": cond,
                            "sigma": sigma,
                            **w_summary,
                        })

                    # Cache test predictions for clean condition (confusion matrix)
                    if split_name == "test" and cond == "normal":
                        cache_key = f"{model_name}_{t_mode}"
                        predictions_cache[cache_key] = {
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }

    # Save degradation results
    results_df = pd.DataFrame(results_records)
    out_csv = RESULTS_DIR / "degradation_results.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Saved degradation evaluation results -> {out_csv}")

    # Save weights summary
    if adaptive_weights_records:
        w_df = pd.DataFrame(adaptive_weights_records)
        w_csv = RESULTS_DIR / "adaptive_fusion_weights.csv"
        w_df.to_csv(w_csv, index=False)
        logger.info(f"Saved adaptive fusion weights -> {w_csv}")

    return results_df, predictions_cache


# ===========================================================================
# 4. Sigma Sweep Routine (Test Set)
# ===========================================================================

def run_sigma_sweep(
    models: List[str],
    train_modes: List[str],
    seed: int = 42,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Evaluates noisy conditions across varying sigma values on the TEST set.

    Noise conditions swept:
        - noisy_acc
        - noisy_gyro
        - noisy_both

    Sigma values: [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    Outputs: results/noise_sweep.csv
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*70}\n  RUNNING SIGMA SWEEP (Test Set)\n{'='*70}")

    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )

    noise_conditions = ["noisy_acc", "noisy_gyro", "noisy_both"]
    sweep_records: List[Dict[str, Any]] = []

    for model_name in models:
        for t_mode in train_modes:
            try:
                model = load_experiment_model(model_name, t_mode, device)
            except FileNotFoundError as err:
                logger.error(f"Cannot run sigma sweep for {model_name} ({t_mode}): {err}")
                continue

            for cond in noise_conditions:
                for sigma in SIGMA_SWEEP_VALUES:
                    metrics, _, _, _ = evaluate_dataset(
                        model=model,
                        loader=test_loader,
                        condition=cond,
                        sigma=sigma,
                        seed=seed,
                        device=device,
                        collect_weights=False,
                    )

                    rec = {
                        "model": model_name,
                        "training_mode": t_mode,
                        "condition": cond,
                        "sigma": float(sigma),
                        "accuracy": metrics["accuracy"],
                        "precision_macro": metrics["precision_macro"],
                        "recall_macro": metrics["recall_macro"],
                        "f1_macro": metrics["f1_macro"],
                        "n_windows": metrics["n_windows"],
                    }
                    sweep_records.append(rec)
                    logger.info(
                        f"[SWEEP] {model_name} ({t_mode}) | {cond:11s} | sigma={sigma:.2f} "
                        f"-> Acc: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_macro']:.4f}"
                    )

    sweep_df = pd.DataFrame(sweep_records)
    out_csv = RESULTS_DIR / "noise_sweep.csv"
    sweep_df.to_csv(out_csv, index=False)
    logger.info(f"Saved noise sweep results -> {out_csv}")
    return sweep_df


# ===========================================================================
# 5. Publication-Quality Visualizations (300 DPI)
# ===========================================================================

def plot_degradation_comparison(
    metric_name: str,
    degradation_csv: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Renders grouped bar chart comparing Baseline vs Adaptive across conditions.
    Two subplots: left = trained_on_normal, right = degradation_aware.
    """
    csv_file = degradation_csv or (RESULTS_DIR / "degradation_results.csv")
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}")
    df = pd.read_csv(csv_file)

    # Filter to test split
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test split records found in degradation_results.csv")

    modes = ["trained_on_normal", "degradation_aware"]
    conditions = ["normal", "noisy_acc", "noisy_gyro", "missing_acc", "missing_gyro", "noisy_both"]
    cond_labels = ["Normal", "Noisy\nAcc", "Noisy\nGyro", "Missing\nAcc", "Missing\nGyro", "Noisy\nBoth"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    metric_col = "accuracy" if metric_name == "accuracy" else "f1_macro"
    y_label = "Test Accuracy (%)" if metric_name == "accuracy" else "Macro F1 Score"
    multiplier = 100.0 if metric_name == "accuracy" else 1.0

    bar_width = 0.35
    x = np.arange(len(conditions))

    color_baseline = "#3182CE"   # Professional blue
    color_adaptive = "#DD6B20"   # Professional amber/orange

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        sub = test_df[test_df["training_mode"] == mode]

        base_vals = []
        adapt_vals = []
        for c in conditions:
            bv = sub[(sub["model"] == "baseline") & (sub["condition"] == c)]
            av = sub[(sub["model"] == "adaptive") & (sub["condition"] == c)]
            base_vals.append(float(bv[metric_col].iloc[0]) * multiplier if not bv.empty else 0.0)
            adapt_vals.append(float(av[metric_col].iloc[0]) * multiplier if not av.empty else 0.0)

        rects1 = ax.bar(x - bar_width / 2, base_vals, bar_width, label="Baseline CNN", color=color_baseline, alpha=0.9, edgecolor="black", lw=0.5)
        rects2 = ax.bar(x + bar_width / 2, adapt_vals, bar_width, label="Adaptive Fusion", color=color_adaptive, alpha=0.9, edgecolor="black", lw=0.5)

        # Value labels on top of bars
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}" if metric_name == "accuracy" else f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=0)

        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}" if metric_name == "accuracy" else f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=0)

        mode_title = "Clean Training (trained_on_normal)" if mode == "trained_on_normal" else "Robust Augmentation (degradation_aware)"
        ax.set_title(mode_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=9.5)
        ax.set_xlabel("Sensor Degradation Condition", fontsize=10.5)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=9.5, loc="lower left")

        if metric_name == "accuracy":
            ax.set_ylim(0, 108)
        else:
            ax.set_ylim(0, 1.08)

    axes[0].set_ylabel(y_label, fontsize=11, fontweight="bold")
    title_metric = "Accuracy" if metric_name == "accuracy" else "Macro F1"
    fig.suptitle(f"Sensor Degradation Robustness: Baseline CNN vs. Adaptive Sensor Fusion ({title_metric})", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = out_path or (RESULTS_DIR / f"degradation_{metric_name}_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {out_file.name} -> {out_file}")
    return out_file


def plot_noise_sweep(
    noise_csv: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Renders 3-panel line plot: Accuracy vs Sigma for noisy_acc, noisy_gyro, and noisy_both.
    """
    csv_file = noise_csv or (RESULTS_DIR / "noise_sweep.csv")
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}")
    df = pd.read_csv(csv_file)

    conditions = ["noisy_acc", "noisy_gyro", "noisy_both"]
    titles = ["Noisy Accelerometer", "Noisy Gyroscope", "Noisy Both (All 12 Channels)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300, sharey=True)

    styles = {
        ("baseline", "trained_on_normal"): {"color": "#3182CE", "ls": "--", "marker": "o", "label": "Baseline (Clean Train)"},
        ("adaptive", "trained_on_normal"): {"color": "#DD6B20", "ls": "-",  "marker": "s", "label": "Adaptive (Clean Train)"},
        ("baseline", "degradation_aware"): {"color": "#3182CE", "ls": ":",  "marker": "^", "label": "Baseline (Deg-Aware Train)"},
        ("adaptive", "degradation_aware"): {"color": "#DD6B20", "ls": "-.", "marker": "d", "label": "Adaptive (Deg-Aware Train)"},
    }

    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        cond_df = df[df["condition"] == cond]

        for (m, t_mode), style in styles.items():
            sub = cond_df[(cond_df["model"] == m) & (cond_df["training_mode"] == t_mode)].sort_values("sigma")
            if not sub.empty:
                ax.plot(
                    sub["sigma"],
                    sub["accuracy"] * 100.0,
                    color=style["color"],
                    linestyle=style["ls"],
                    marker=style["marker"],
                    linewidth=2.0,
                    markersize=6,
                    label=style["label"],
                )

        ax.set_title(titles[idx], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Noise Std Dev (σ in standardized units)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=True, fontsize=8.5, loc="lower left")
    fig.suptitle("Degradation Sweep: Accuracy Trajectory across Noise Magnitudes (σ = 0 to 2.0)", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = out_path or (RESULTS_DIR / "noise_sweep.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved noise_sweep.png -> {out_file}")
    return out_file


def plot_confusion_matrices(
    predictions_cache: Dict[str, Any],
    label_mapping_path: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Renders high-resolution confusion matrices for Baseline and Adaptive on clean test set.
    """
    mapping_file = label_mapping_path or DEFAULT_LABEL_MAPPING_PATH
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names = [name for name, _ in sorted(mapping.items(), key=lambda x: x[1])]

    files: List[Optional[Path]] = []

    for model_name in ["baseline", "adaptive"]:
        key = f"{model_name}_trained_on_normal"
        if key not in predictions_cache:
            # Try to get from disk if not in cache
            logger.warning(f"No prediction cache for {key}. Skipping CM plot.")
            files.append(None)
            continue

        data = predictions_cache[key]
        y_true = data["y_true"]
        y_pred = data["y_pred"]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        cmap = plt.cm.Blues if model_name == "baseline" else plt.cm.Oranges

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (Row-Normalized)")

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9.5)
        ax.set_yticklabels(class_names, fontsize=9.5)

        # Annotate cells with counts and percentages
        thresh = cm_norm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                raw_val = cm[i, j]
                norm_val = cm_norm[i, j]
                text_color = "white" if norm_val > thresh else "black"
                ax.text(
                    j, i, f"{raw_val}\n({norm_val*100:.1f}%)",
                    ha="center", va="center",
                    color=text_color, fontsize=8,
                )

        disp_title = "Baseline 1D CNN" if model_name == "baseline" else "Adaptive Sensor Fusion"
        ax.set_title(f"{disp_title} - Test Confusion Matrix (Clean Data)", fontsize=11.5, fontweight="bold", pad=12)
        ax.set_ylabel("True Activity", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Predicted Activity", fontsize=10.5, fontweight="bold")
        plt.tight_layout()

        out_file = RESULTS_DIR / f"confusion_matrix_normal_{model_name}.png"
        plt.savefig(out_file, dpi=300)
        plt.close(fig)
        logger.info(f"Saved confusion matrix -> {out_file}")
        files.append(out_file)

    return files[0], files[1]


def plot_adaptive_fusion_weights(
    weights_csv: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Renders bar chart of mean w_acc vs w_gyro per condition for AdaptiveSensorFusion.
    Shows the dynamic weight reallocation under sensor faults.
    """
    csv_file = weights_csv or (RESULTS_DIR / "adaptive_fusion_weights.csv")
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}")
    df = pd.read_csv(csv_file)

    conditions = ["normal", "noisy_acc", "noisy_gyro", "missing_acc", "missing_gyro", "noisy_both"]
    cond_labels = ["Normal", "Noisy\nAcc", "Noisy\nGyro", "Missing\nAcc", "Missing\nGyro", "Noisy\nBoth"]

    modes = df["training_mode"].unique().tolist()
    n_modes = len(modes)

    fig, axes = plt.subplots(1, max(1, n_modes), figsize=(6.5 * max(1, n_modes), 5), dpi=300, squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[0, idx]
        sub = df[df["training_mode"] == mode]

        w_acc_vals = []
        w_gyro_vals = []
        for c in conditions:
            row = sub[sub["condition"] == c]
            if not row.empty:
                w_acc_vals.append(float(row["w_acc_mean"].iloc[0]))
                w_gyro_vals.append(float(row["w_gyro_mean"].iloc[0]))
            else:
                w_acc_vals.append(0.5)
                w_gyro_vals.append(0.5)

        x = np.arange(len(conditions))
        bar_width = 0.35

        rects1 = ax.bar(x - bar_width / 2, w_acc_vals, bar_width, label="w_acc (Accelerometer Branch)", color="#3182CE", alpha=0.9, edgecolor="black", lw=0.5)
        rects2 = ax.bar(x + bar_width / 2, w_gyro_vals, bar_width, label="w_gyro (Gyro/Attitude Branch)", color="#38A169", alpha=0.9, edgecolor="black", lw=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        mode_title = "Clean Training (trained_on_normal)" if mode == "trained_on_normal" else "Robust Augmentation (degradation_aware)"
        ax.set_title(f"Adaptive Fusion Weights ({mode_title})", fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=9.5)
        ax.set_xlabel("Sensor Degradation Condition", fontsize=10)
        ax.set_ylabel("Mean Fusion Weight", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=9, loc="upper right")

    fig.suptitle("Adaptive Sensor Fusion: Dynamic Branch Weight Allocation across Conditions", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = out_path or (RESULTS_DIR / "adaptive_fusion_weights.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved adaptive_fusion_weights.png -> {out_file}")
    return out_file


# ===========================================================================
# 6. Factual Markdown Summary
# ===========================================================================

def print_factual_summary() -> None:
    """
    Parses results/degradation_results.csv and prints a strict, objective
    Markdown summary table. Completely truthful to the numbers.
    """
    deg_csv = RESULTS_DIR / "degradation_results.csv"
    if not deg_csv.exists():
        print("No degradation_results.csv found.")
        return

    df = pd.read_csv(deg_csv)
    test_df = df[df["split"] == "test"]

    print("\n" + "=" * 80)
    print("  OBJECTIVE EXPERIMENTAL SUMMARY: BASELINE CNN vs. ADAPTIVE SENSOR FUSION")
    print("  MotionSense Dataset (Test Split: 4,613 temporal windows, 4 unseen participants)")
    print("=" * 80)

    for mode in ["trained_on_normal", "degradation_aware"]:
        sub = test_df[test_df["training_mode"] == mode]
        mode_label = "CLEAN TRAINING (trained_on_normal)" if mode == "trained_on_normal" else "ROBUST AUGMENTATION (degradation_aware)"
        print(f"\n### {mode_label}\n")

        print(f"| {'Condition':<15} | {'Baseline Acc (%)':<16} | {'Adaptive Acc (%)':<16} | {'Delta Acc (%)':<14} | {'Baseline F1':<12} | {'Adaptive F1':<12} | {'Superior Model':<15} |")
        print(f"|{'-'*17}|{'-'*18}|{'-'*18}|{'-'*16}|{'-'*14}|{'-'*14}|{'-'*17}|")

        for cond in ["normal", "noisy_acc", "noisy_gyro", "missing_acc", "missing_gyro", "noisy_both"]:
            b_row = sub[(sub["model"] == "baseline") & (sub["condition"] == cond)]
            a_row = sub[(sub["model"] == "adaptive") & (sub["condition"] == cond)]

            if not b_row.empty and not a_row.empty:
                b_acc = float(b_row["accuracy"].iloc[0]) * 100.0
                a_acc = float(a_row["accuracy"].iloc[0]) * 100.0
                delta = a_acc - b_acc

                b_f1 = float(b_row["f1_macro"].iloc[0])
                a_f1 = float(a_row["f1_macro"].iloc[0])

                if abs(delta) < 0.1:
                    sup = "Parity (≈)"
                elif delta > 0:
                    sup = f"Adaptive (+{delta:.2f}%)"
                else:
                    sup = f"Baseline ({delta:.2f}%)"

                print(f"| {cond:<15} | {b_acc:>14.2f}% | {a_acc:>14.2f}% | {delta:>+12.2f}% | {b_f1:>10.4f} | {a_f1:>10.4f} | {sup:<15} |")

    # Print weights summary if available
    w_csv = RESULTS_DIR / "adaptive_fusion_weights.csv"
    if w_csv.exists():
        w_df = pd.read_csv(w_csv)
        print("\n\n### Adaptive Fusion Dynamic Branch Weights (Test Split)\n")
        print(f"| {'Training Mode':<22} | {'Condition':<15} | {'Mean w_acc':<12} | {'Mean w_gyro':<12} | {'Behavior Description':<30} |")
        print(f"|{'-'*24}|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*32}|")

        for _, r in w_df.iterrows():
            w_acc = float(r["w_acc_mean"])
            w_gyro = float(r["w_gyro_mean"])
            c = str(r["condition"])
            m = str(r["training_mode"])

            if w_acc > 0.8:
                desc = "Shifted heavily to Accelerometer"
            elif w_gyro > 0.8:
                desc = "Shifted heavily to Gyroscope"
            elif abs(w_acc - w_gyro) < 0.1:
                desc = "Balanced multi-modal reliance"
            elif w_acc > w_gyro:
                desc = "Favoring Accelerometer branch"
            else:
                desc = "Favoring Gyroscope branch"

            print(f"| {m:<22} | {c:<15} | {w_acc:>10.4f} | {w_gyro:>10.4f} | {desc:<30} |")

    print("\n" + "=" * 80 + "\n")


# ===========================================================================
# 7. CLI Orchestrator
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete HAR experiment pipeline (Member 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (train, eval, sweep, plot, summary)")
    parser.add_argument("--train", action="store_true", help="Train models with selected hyperparameters")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing checkpoints")
    parser.add_argument("--eval-degradation", action="store_true", help="Run 6-condition degradation evaluation")
    parser.add_argument("--sigma-sweep", action="store_true", help="Run sigma sweep over noise standard deviation")
    parser.add_argument("--plot", action="store_true", help="Generate all publication-quality figures")
    parser.add_argument("--summary", action="store_true", help="Print objective summary table")
    parser.add_argument("--models", default="both", help="'baseline', 'adaptive', or 'both'")
    parser.add_argument("--train-modes", default="both", help="'trained_on_normal', 'degradation_aware', or 'both'")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds (e.g. '42' or '42,43,44')")
    parser.add_argument("--smoke", action="store_true", help="Run fast verification on small subsets")
    parser.add_argument("--smoke-epochs", type=int, default=2, help="Epoch cap for smoke mode")
    parser.add_argument("--smoke-samples", type=int, default=256, help="Samples cap for smoke mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse models
    raw_models = args.models.lower().strip()
    models = ["baseline", "adaptive"] if raw_models == "both" else [m.strip() for m in raw_models.split(",") if m.strip()]

    # Parse training modes
    raw_modes = args.train_modes.lower().strip()
    if raw_modes == "both":
        train_modes = ["trained_on_normal", "degradation_aware"]
    elif raw_modes in ("normal", "trained_on_normal"):
        train_modes = ["trained_on_normal"]
    elif raw_modes in ("degradation_aware", "deg_aware"):
        train_modes = ["degradation_aware"]
    else:
        raise ValueError(f"Unknown training mode: {raw_modes}")

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # If --all or no action specified, set all to True
    run_all = args.all or not (args.train or args.eval_degradation or args.sigma_sweep or args.plot or args.summary)

    should_train = (run_all or args.train) and not args.skip_train
    should_eval_deg = run_all or args.eval_degradation
    should_sweep = run_all or args.sigma_sweep
    should_plot = run_all or args.plot
    should_summary = run_all or args.summary

    logger.info(
        f"Experiment Orchestrator initialized:\n"
        f"  - models: {models}\n"
        f"  - train_modes: {train_modes}\n"
        f"  - seeds: {seeds}\n"
        f"  - should_train: {should_train}\n"
        f"  - should_eval_degradation: {should_eval_deg}\n"
        f"  - should_sigma_sweep: {should_sweep}\n"
        f"  - should_plot: {should_plot}\n"
        f"  - smoke: {args.smoke}"
    )

    pred_cache: Dict[str, Any] = {}

    # Step 1: Final Training
    if should_train:
        logger.info("\n>>> STEP 1: FINAL TRAINING <<<")
        run_final_training(
            models=models,
            train_modes=train_modes,
            seeds=seeds,
            smoke=args.smoke,
            smoke_epochs=args.smoke_epochs,
            smoke_samples=args.smoke_samples,
        )

    # Step 2: Degradation Evaluation
    if should_eval_deg:
        logger.info("\n>>> STEP 2: DEGRADATION EVALUATION <<<")
        _, pred_cache = run_degradation_evaluation(
            models=models,
            train_modes=train_modes,
            seed=seeds[0],
        )

    # Step 3: Sigma Sweep
    if should_sweep:
        logger.info("\n>>> STEP 3: SIGMA SWEEP <<<")
        run_sigma_sweep(
            models=models,
            train_modes=train_modes,
            seed=seeds[0],
        )

    # Step 4: Plots
    if should_plot:
        logger.info("\n>>> STEP 4: PLOT GENERATION <<<")
        try:
            plot_degradation_comparison("accuracy")
            plot_degradation_comparison("f1")
            plot_noise_sweep()
            plot_confusion_matrices(pred_cache)
            plot_adaptive_fusion_weights()
        except Exception as exc:
            logger.error(f"Error during plot generation: {exc}", exc_info=True)

    # Step 5: Summary
    if should_summary:
        logger.info("\n>>> STEP 5: FACTUAL SUMMARY <<<")
        print_factual_summary()

    logger.info("All requested pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
