#!.venv/Scripts/python.exe
"""
src/train.py

Phase 3/4 Training & Validation Pipeline for Human Activity Recognition (MotionSense).
Owned by Member 2.

Key Features:
    1. Model Factory (build_model):
       Instantiates BaselineCNN or AdaptiveSensorFusion directly from src packages.
    2. Training Engine (train_model):
       - CrossEntropyLoss with Adam or SGD (momentum=0.9).
       - Per-epoch training & validation evaluation (validation strictly on clean data).
       - Early stopping on validation loss with state restoration of best weights.
       - Optional ReduceLROnPlateau scheduler and gradient clipping.
       - Explicit all-ones masks for AdaptiveSensorFusion in normal mode so train/eval paths match.
       - Optional degradation-aware training mode using degrade_batch_random().
    3. Artifacts & Logging:
       - Saves best model checkpoints to models/baseline_best.pth and models/adaptive_best.pth.
       - Saves comprehensive model configuration and validation metrics to models/<name>_config.json.
       - Appends per-epoch metrics to results/training_history.csv (keyed by unique run_id).
       - Saves per-run metrics summary to results/metrics_<run_id>.json.
    4. Visualization (plot_curves):
       Generates publication-quality 300 DPI loss and accuracy curves (results/loss_curve.png,
       results/accuracy_curve.png) strictly parsed from the saved history CSV.
    5. Smoke Test Mode:
       Rapid 2-epoch validation on a subset to verify end-to-end execution without polluting
       production result directories.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set temporary cache dir for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_har_train"))

# Add project root to sys.path
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
from torch.utils.data import DataLoader, Subset

from src.data_loader import (
    DEFAULT_CSV_PATH,
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SUMMARY_PATH,
    get_dataloaders,
)
from src.model import (
    ACTIVITY_CLASSES,
    BaselineCNN,
    NUM_ACTIVITIES,
    NUM_SENSOR_FEATURES,
    SENSOR_COLUMNS,
    save_model,
)
from src.sensor_degradation import (
    ACC_INDICES,
    GYRO_INDICES,
    degrade_batch_random,
)
from src.sensor_fusion import (
    AdaptiveSensorFusion,
    save_adaptive_model,
)

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ============================================================================
# 1. Model Factory & Uniform Dropout Override Helper
# ============================================================================
def set_uniform_dropout(model: nn.Module, p: float) -> nn.Module:
    """
    Sets the dropout probability 'p' of every nn.Dropout and nn.Dropout1d layer
    in the model after initialization.

    NOTE: This post-initialization traversal overrides Member 1's default
    hardcoded values (dropout_rate=0.2, dense_dropout=0.3) without modifying
    Member 1's source code in src/model.py or src/sensor_fusion.py.

    Args:
        model: PyTorch model instance (BaselineCNN or AdaptiveSensorFusion).
        p: Target dropout probability in [0.0, 1.0).

    Returns:
        The mutated model instance with updated dropout layers.
    """
    if not (0.0 <= p < 1.0):
        raise ValueError(f"Dropout probability must be in [0.0, 1.0), but got {p}")

    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d)):
            module.p = float(p)
            count += 1

    return model


def get_model_dropouts(model: nn.Module) -> List[float]:
    """
    Inspects and returns a list of dropout probabilities from all nn.Dropout
    and nn.Dropout1d layers in the model.
    """
    return [
        float(module.p)
        for module in model.modules()
        if isinstance(module, (nn.Dropout, nn.Dropout1d))
    ]


def build_model(
    name: str,
    dropout: Optional[float] = None,
    num_classes: int = NUM_ACTIVITIES,
) -> nn.Module:
    """
    Constructs a model instance with configurable dropout.

    Args:
        name: Architecture name ('baseline' or 'adaptive').
        dropout: Optional uniform dropout probability. If specified, overrides
                 Member 1's default values across all nn.Dropout layers using
                 set_uniform_dropout().
        num_classes: Number of activity target classes (default 6).

    Returns:
        Instantiated and configured nn.Module.
    """
    name_clean = name.lower().strip()

    # Member 1's models are instantiated with their standard default constructors
    if name_clean == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    elif name_clean == "adaptive":
        model = AdaptiveSensorFusion(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{name}'. Expected 'baseline' or 'adaptive'.")

    # Override Member 1's default values post-init without modifying their source files
    if dropout is not None:
        set_uniform_dropout(model, dropout)

    return model


# ============================================================================
# 2. Training Engine
# ============================================================================
def train_model(
    model_name: str,
    train_condition: str = "normal",
    lr: float = 1e-3,
    batch_size: int = 64,
    dropout: Optional[float] = None,
    max_epochs: int = 40,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    seed: int = 42,
    patience: int = 7,
    use_scheduler: bool = False,
    grad_clip: Optional[float] = None,
    output_tag: Optional[str] = None,
    results_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    smoke_test: bool = False,
    smoke_samples: int = 256,
) -> Dict[str, Any]:
    """
    Executes a complete training and validation cycle with early stopping.

    Args:
        model_name: 'baseline' or 'adaptive'.
        train_condition: 'normal' or 'degradation_aware'.
        lr: Learning rate.
        batch_size: DataLoader mini-batch size.
        dropout: Configurable dropout rate.
        max_epochs: Maximum training epochs.
        optimizer_name: 'adam' or 'sgd' (SGD with momentum=0.9).
        weight_decay: L2 regularization penalty.
        seed: Random seed for data shuffling and model initialization.
        patience: Early stopping patience epochs on validation loss.
        use_scheduler: If True, uses ReduceLROnPlateau.
        grad_clip: Optional max norm for gradient clipping (e.g. 1.0).
        output_tag: Optional tag for run identification.
        results_dir: Custom results directory (defaults to PROJECT_ROOT/results).
        models_dir: Custom models directory (defaults to PROJECT_ROOT/models).
        smoke_test: If True, runs a rapid 2-epoch subset verification.
        smoke_samples: Number of train/val samples used during smoke test.

    Returns:
        Dictionary detailing best validation performance and run metadata.
    """
    start_wall_time = time.time()

    # 1. Device Selection & Seeding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tag_str = f"_{output_tag}" if output_tag else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{train_condition}{tag_str}_{timestamp}"

    out_res_dir = results_dir or RESULTS_DIR
    out_mod_dir = models_dir or MODELS_DIR
    out_res_dir.mkdir(parents=True, exist_ok=True)
    out_mod_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Loaders
    logger.info(f"[{run_id}] Loading MotionSense data loaders (seed={seed}, batch_size={batch_size})...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )

    # If smoke test, subsample loaders to run in seconds
    if smoke_test:
        max_epochs = min(max_epochs, 2)
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset
        n_tr = min(smoke_samples, len(train_ds))
        n_va = min(smoke_samples // 2, len(val_ds))
        train_loader = DataLoader(Subset(train_ds, list(range(n_tr))), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_ds, list(range(n_va))), batch_size=batch_size, shuffle=False)
        logger.info(f"[SMOKE TEST] Truncated dataset to {n_tr} train and {n_va} val samples, max_epochs={max_epochs}")

    # 3. Model Instantiation
    model = build_model(model_name, dropout=dropout).to(device)
    param_counts = model.count_parameters()
    logger.info(
        f"[{run_id}] Initialized {model.__class__.__name__} on device: {device} | "
        f"Total Parameters: {param_counts['total_parameters']:,} "
        f"(Trainable: {param_counts['trainable_parameters']:,})"
    )

    # 4. Optimizer, Criterion & Scheduler
    criterion = nn.CrossEntropyLoss()

    opt_name_clean = optimizer_name.lower().strip()
    if opt_name_clean == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name_clean == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Expected 'adam' or 'sgd'.")

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        if use_scheduler
        else None
    )

    # 5. Training State Tracking
    history_records: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch_loss = -1
    best_epoch_acc = -1
    best_weights: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    is_adaptive = isinstance(model, AdaptiveSensorFusion)

    logger.info(f"[{run_id}] Beginning training loop for {max_epochs} epochs (Patience={patience})...")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # --------------------------------------------------------------------
        # Training Phase
        # --------------------------------------------------------------------
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            b_size = batch_x.size(0)

            # Apply training condition
            if train_condition == "degradation_aware":
                # Stochastic augmentation during training
                batch_x_aug, masks = degrade_batch_random(
                    batch_x, p_noise=0.3, p_missing=0.3, sigma=0.5
                )
                if is_adaptive:
                    logits = model(
                        batch_x_aug,
                        mask_acc=masks["mask_acc"],
                        mask_gyro=masks["mask_gyro"],
                    )
                else:
                    logits = model(batch_x_aug)
            else:
                # Normal condition: explicit all-ones masks for Adaptive model
                if is_adaptive:
                    mask_ones = torch.ones(b_size, dtype=torch.float32, device=device)
                    logits = model(batch_x, mask_acc=mask_ones, mask_gyro=mask_ones)
                else:
                    logits = model(batch_x)

            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            running_train_loss += loss.item() * b_size
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += b_size

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # --------------------------------------------------------------------
        # Validation Phase (Always on clean normal data)
        # --------------------------------------------------------------------
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                b_size = batch_x.size(0)

                if is_adaptive:
                    mask_ones = torch.ones(b_size, dtype=torch.float32, device=device)
                    logits = model(batch_x, mask_acc=mask_ones, mask_gyro=mask_ones)
                else:
                    logits = model(batch_x)

                loss = criterion(logits, batch_y)
                running_val_loss += loss.item() * b_size
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += b_size

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Track best validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch_acc = epoch

        # Check early stopping on validation loss
        is_best_loss = epoch_val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = epoch_val_loss
            best_epoch_loss = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_marker = " [BEST]"
        else:
            patience_counter += 1
            best_marker = f" (Patience: {patience_counter}/{patience})"

        record = {
            "model": model_name,
            "condition": train_condition,
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 5),
            "train_acc": round(epoch_train_acc, 5),
            "val_loss": round(epoch_val_loss, 5),
            "val_acc": round(epoch_val_acc, 5),
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 2),
        }
        history_records.append(record)

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.2f}% | "
            f"Time: {epoch_time:.1f}s{best_marker}"
        )

        if patience_counter >= patience and not smoke_test:
            logger.info(f"[{run_id}] Early stopping triggered at epoch {epoch} (Best val loss: {best_val_loss:.4f})")
            break

    # 6. Restore Best Model Weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info(f"[{run_id}] Restored best model weights from epoch {best_epoch_loss}.")

    total_wall_time = time.time() - start_wall_time
    history_df = pd.DataFrame(history_records)

    # 7. Checkpoint & Artifact Serialization
    checkpoint_file = out_mod_dir / f"{model_name}_best.pth"
    config_file = out_mod_dir / f"{model_name}_config.json"
    metrics_file = out_res_dir / f"metrics_{run_id}.json"
    history_csv = out_res_dir / "training_history.csv"

    # Save model checkpoint using Member 1's helpers
    if not smoke_test:
        if is_adaptive:
            save_adaptive_model(model, str(checkpoint_file))
        else:
            save_model(model, str(checkpoint_file))
        logger.info(f"Saved best model checkpoint to: {checkpoint_file}")

    # Build comprehensive config record
    config_data = {
        "model_name": model_name,
        "run_id": run_id,
        "train_condition": train_condition,
        "hyperparameters": {
            "lr": lr,
            "batch_size": batch_size,
            "dropout": dropout if dropout is not None else 0.2,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "use_scheduler": use_scheduler,
            "grad_clip": grad_clip,
        },
        "dataset_config": {
            "window_size": 100,
            "stride": 50,
            "feature_order": SENSOR_COLUMNS,
            "acc_indices": ACC_INDICES,
            "gyro_indices": GYRO_INDICES,
            "num_features": NUM_SENSOR_FEATURES,
            "num_classes": NUM_ACTIVITIES,
            "scaler_path": str(DEFAULT_SCALER_PATH),
            "label_mapping_path": str(DEFAULT_LABEL_MAPPING_PATH),
            "split_file_path": str(DEFAULT_SPLIT_PATH),
        },
        "performance": {
            "best_epoch_val_loss": best_epoch_loss,
            "best_val_loss": round(best_val_loss, 5),
            "best_epoch_val_acc": best_epoch_acc,
            "best_val_acc": round(best_val_acc, 5),
            "total_epochs_trained": len(history_records),
            "total_wall_time_s": round(total_wall_time, 2),
        },
        "hardware": {
            "device": str(device),
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
        },
        "timestamp_utc": timestamp,
    }

    if not smoke_test:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Saved model configuration to: {config_file}")

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Saved run metrics summary to: {metrics_file}")

        # Append to master history CSV without overwriting previous runs
        if history_csv.exists():
            existing_df = pd.read_csv(history_csv)
            combined_df = pd.concat([existing_df, history_df], ignore_index=True)
            combined_df.to_csv(history_csv, index=False)
        else:
            history_df.to_csv(history_csv, index=False)
        logger.info(f"Appended epoch history to: {history_csv}")

    return {
        "model": model,
        "history_df": history_df,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch_loss,
        "run_id": run_id,
        "config_data": config_data,
    }


# ============================================================================
# 3. Curve Visualization Engine
# ============================================================================
def plot_curves(
    history_df: Optional[pd.DataFrame] = None,
    history_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Renders publication-quality Loss and Accuracy curves directly from history records.

    Strictly reads and parses real values from history records or results/training_history.csv.
    """
    destination_dir = out_dir or RESULTS_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)

    if history_df is None:
        csv_file = history_csv or (destination_dir / "training_history.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"Training history CSV not found at: {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        df = history_df.copy()

    loss_png = destination_dir / "loss_curve.png"
    acc_png = destination_dir / "accuracy_curve.png"

    # Identify models present in history
    models_present = df["model"].unique().tolist()
    n_models = len(models_present)

    # ------------------------------------------------------------------------
    # Plot 1: Loss Curve
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, max(1, n_models), figsize=(6 * max(1, n_models), 5), dpi=300, squeeze=False)

    for idx, m_name in enumerate(models_present):
        ax = axes[0, idx]
        # Filter latest run for this model
        sub_df = df[df["model"] == m_name]
        latest_run = sub_df["run_id"].iloc[-1]
        run_df = sub_df[sub_df["run_id"] == latest_run].sort_values("epoch")

        epochs = run_df["epoch"]
        ax.plot(epochs, run_df["train_loss"], label="Train Loss", color="#3182CE", lw=2, marker="o", ms=4)
        ax.plot(epochs, run_df["val_loss"], label="Val Loss", color="#E53E3E", lw=2, marker="s", ms=4)

        # Mark minimum validation loss
        min_val_idx = run_df["val_loss"].idxmin()
        best_ep = run_df.loc[min_val_idx, "epoch"]
        best_vl = run_df.loc[min_val_idx, "val_loss"]
        ax.scatter([best_ep], [best_vl], color="#E53E3E", s=100, zorder=5)
        ax.annotate(
            f"Min Val Loss: {best_vl:.4f}\n(Epoch {int(best_ep)})",
            xy=(best_ep, best_vl),
            xytext=(best_ep + 0.5, best_vl + 0.15),
            arrowprops=dict(arrowstyle="->", color="#E53E3E", lw=1.2),
            fontsize=9,
            fontweight="bold",
            color="#9B2C2C",
        )

        display_name = "Baseline 1D CNN" if m_name == "baseline" else "Adaptive Sensor Fusion"
        ax.set_title(f"{display_name} - Loss Trajectory", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("CrossEntropy Loss", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(loss_png, dpi=300)
    plt.close(fig)
    logger.info(f"Saved loss curve to: {loss_png}")

    # ------------------------------------------------------------------------
    # Plot 2: Accuracy Curve
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, max(1, n_models), figsize=(6 * max(1, n_models), 5), dpi=300, squeeze=False)

    for idx, m_name in enumerate(models_present):
        ax = axes[0, idx]
        sub_df = df[df["model"] == m_name]
        latest_run = sub_df["run_id"].iloc[-1]
        run_df = sub_df[sub_df["run_id"] == latest_run].sort_values("epoch")

        epochs = run_df["epoch"]
        ax.plot(epochs, run_df["train_acc"] * 100, label="Train Accuracy", color="#38A169", lw=2, marker="o", ms=4)
        ax.plot(epochs, run_df["val_acc"] * 100, label="Val Accuracy", color="#805AD5", lw=2, marker="s", ms=4)

        max_val_idx = run_df["val_acc"].idxmax()
        best_ep = run_df.loc[max_val_idx, "epoch"]
        best_va = run_df.loc[max_val_idx, "val_acc"] * 100
        ax.scatter([best_ep], [best_va], color="#805AD5", s=100, zorder=5)
        ax.annotate(
            f"Max Val Acc: {best_va:.2f}%\n(Epoch {int(best_ep)})",
            xy=(best_ep, best_va),
            xytext=(best_ep + 0.5, best_va - 4.0),
            arrowprops=dict(arrowstyle="->", color="#805AD5", lw=1.2),
            fontsize=9,
            fontweight="bold",
            color="#553C9A",
        )

        display_name = "Baseline 1D CNN" if m_name == "baseline" else "Adaptive Sensor Fusion"
        ax.set_title(f"{display_name} - Accuracy Trajectory", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(bottom=0, top=105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(acc_png, dpi=300)
    plt.close(fig)
    logger.info(f"Saved accuracy curve to: {acc_png}")

    return loss_png, acc_png


# ============================================================================
# 4. Dropout Verification & Smoke Test Routines
# ============================================================================
def verify_dropout_configuration() -> None:
    """
    Demonstrates and verifies that set_uniform_dropout() correctly updates
    all nn.Dropout layers for both BaselineCNN and AdaptiveSensorFusion across
    p in [0.2, 0.3, 0.5].
    Prints the dropout values inside each model to prove that Member 1's
    defaults are overridden.
    """
    print("=" * 78)
    print("  VERIFYING UNIFORM DROPOUT CONFIGURATION (set_uniform_dropout)")
    print("=" * 78)

    test_dropouts = [0.2, 0.3, 0.5]

    for model_name in ["baseline", "adaptive"]:
        display_name = "BaselineCNN" if model_name == "baseline" else "AdaptiveSensorFusion"
        print(f"\n--- Model: {display_name} ---")

        # 1. Show Member 1's default values when dropout=None
        default_model = build_model(model_name, dropout=None)
        default_dropouts = get_model_dropouts(default_model)
        print(f"  [Default Constructor] (dropout=None):")
        print(f"    Member 1's default dropout values: {default_dropouts}")
        print(f"    Total nn.Dropout layers: {len(default_dropouts)}")

        # 2. Test dropout = 0.2, 0.3, 0.5
        for p in test_dropouts:
            configured_model = build_model(model_name, dropout=p)
            actual_dropouts = get_model_dropouts(configured_model)
            print(f"  [Configured with dropout={p}]:")
            print(f"    Overridden dropout values inside model: {actual_dropouts}")
            assert len(actual_dropouts) == len(default_dropouts), "Layer count changed!"
            assert all(abs(d - p) < 1e-6 for d in actual_dropouts), (
                f"Dropout mismatch! Expected all {p}, but got {actual_dropouts}"
            )
            print(f"    -> Proved: all {len(actual_dropouts)} layers successfully overridden to {p}")

    print("\n" + "=" * 78)
    print("  DROPOUT OVERRIDE PROOF COMPLETE - ALL CHECKS PASSED!")
    print("=" * 78)


def run_smoke_test() -> None:
    """
    Executes a rapid 2-epoch subset verification for both models.

    Strict Isolation:
        Writes exclusively to a temporary sandbox directory to prevent
        modifying or polluting production results/ or models/ folders.
    """
    # First verify and prove dropout override
    verify_dropout_configuration()

    print("\n" + "=" * 78)
    print("  RUNNING src/train.py RAPID SMOKE TEST (2 Epochs on Subset)")
    print("  NOTE: Results are written to a temporary folder and will not")
    print("        pollute production results/ or models/ directories.")
    print("=" * 78)

    with tempfile.TemporaryDirectory(prefix="har_smoke_test_") as tmp_dir:
        temp_path = Path(tmp_dir)
        temp_results = temp_path / "results"
        temp_models = temp_path / "models"

        # 1. Smoke test for BaselineCNN
        print("\n[Smoke Test 1/2] Baseline 1D CNN...")
        res_baseline = train_model(
            model_name="baseline",
            train_condition="normal",
            lr=1e-3,
            batch_size=32,
            dropout=0.2,
            max_epochs=2,
            optimizer_name="adam",
            seed=42,
            patience=2,
            output_tag="smoke",
            results_dir=temp_results,
            models_dir=temp_models,
            smoke_test=True,
            smoke_samples=128,
        )
        assert len(res_baseline["history_df"]) == 2
        print(f"  -> Baseline CNN Smoke Test Passed (Val Acc: {res_baseline['best_val_acc']*100:.2f}%)")

        # 2. Smoke test for AdaptiveSensorFusion
        print("\n[Smoke Test 2/2] Adaptive Sensor Fusion Model...")
        res_adaptive = train_model(
            model_name="adaptive",
            train_condition="normal",
            lr=1e-3,
            batch_size=32,
            dropout=0.2,
            max_epochs=2,
            optimizer_name="adam",
            seed=42,
            patience=2,
            output_tag="smoke",
            results_dir=temp_results,
            models_dir=temp_models,
            smoke_test=True,
            smoke_samples=128,
        )
        assert len(res_adaptive["history_df"]) == 2
        print(f"  -> Adaptive Fusion Smoke Test Passed (Val Acc: {res_adaptive['best_val_acc']*100:.2f}%)")

        # 3. Test Plotting from Temporary History CSV
        print("\n[Smoke Test 3/3] Curve Generation from Temporary History...")
        combined_smoke_df = pd.concat([res_baseline["history_df"], res_adaptive["history_df"]], ignore_index=True)
        temp_history_csv = temp_results / "training_history.csv"
        temp_results.mkdir(parents=True, exist_ok=True)
        combined_smoke_df.to_csv(temp_history_csv, index=False)

        loss_p, acc_p = plot_curves(history_csv=temp_history_csv, out_dir=temp_results)
        assert loss_p.exists()
        assert acc_p.exists()
        print(f"  -> Generated smoke loss curve: {loss_p.name} ({loss_p.stat().st_size:,} bytes)")
        print(f"  -> Generated smoke accuracy curve: {acc_p.name} ({acc_p.stat().st_size:,} bytes)")

    print("\n" + "=" * 78)
    print("  ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("  Production results/ and models/ directories remain completely clean.")
    print("=" * 78)


# ============================================================================
# 5. CLI Entrypoint
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="MotionSense Model Training Pipeline (Member 2)")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["baseline", "adaptive", "both"],
        help="Target architecture to train (default: both)",
    )
    parser.add_argument(
        "--train-condition",
        type=str,
        default="normal",
        choices=["normal", "degradation_aware"],
        help="Training mode: 'normal' or 'degradation_aware' (default: normal)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability override")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs (default: 40)")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer type (default: adam)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--use-scheduler", action="store_true", help="Enable ReduceLROnPlateau scheduler")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping max norm (default: None)")
    parser.add_argument("--output-tag", type=str, default=None, help="Custom tag appended to run_id")
    parser.add_argument(
        "--verify-dropout",
        action="store_true",
        help="Demonstrate and verify that set_uniform_dropout overrides dropouts to 0.2, 0.3, and 0.5",
    )
    parser.add_argument("--smoke", action="store_true", help="Execute rapid 2-epoch smoke test in isolated temp dir")

    args = parser.parse_args()

    if args.verify_dropout:
        verify_dropout_configuration()
        return

    if args.smoke:
        run_smoke_test()
        return

    models_to_train = ["baseline", "adaptive"] if args.model == "both" else [args.model]

    for m in models_to_train:
        logger.info(f"Starting production training run for model: {m.upper()}")
        train_model(
            model_name=m,
            train_condition=args.train_condition,
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            max_epochs=args.epochs,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            seed=args.seed,
            patience=args.patience,
            use_scheduler=args.use_scheduler,
            grad_clip=args.grad_clip,
            output_tag=args.output_tag,
        )

    # Re-plot curves from aggregated history CSV
    plot_curves()


if __name__ == "__main__":
    main()

