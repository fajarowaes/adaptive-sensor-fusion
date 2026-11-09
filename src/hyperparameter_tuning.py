#!.venv/Scripts/python.exe
"""
src/hyperparameter_tuning.py

Phase 3/4 Hyperparameter Tuning for Human Activity Recognition (MotionSense).
Owned by Member 2.

Search Space:
    lr            : [0.001, 0.0005, 0.0001]
    batch_size    : [32, 64, 128]
    dropout       : [0.2, 0.3, 0.5]
    max_epochs    : [20, 30, 50]
    optimizer     : ["adam", "sgd"]
    Full grid     : 3 × 3 × 3 × 3 × 2 = 162 configs per model

Modes:
    staged  : Stage A — 18 configs (lr × batch_size × optimizer, dropout=0.3, max_epochs=30);
              Stage B — top-3 from Stage A × dropout × max_epochs (3×3=9 each = 27 runs) per model.
    full    : Enumerate all 162 configs.
    random  : (DEFAULT) Stratified random sample of N configs (default N=20) so every
              value of every hyperparameter appears at least once in the sample.

Crash-Safety:
    Each CSV row is written immediately (append mode) after its run finishes.
    On restart, existing rows are loaded into an in-memory set of keys; matching
    configs are skipped in O(1) without re-reading the CSV on every call.

Output Artifacts:
    results/hyperparameter_results.csv      — Full run log
    results/best_hyperparameters.json       — Best config per model (never hardcoded)
    results/hyperparameter_summary.png      — Val-acc heatmap by lr × batch_size per optimizer

Hard Rules:
    1. Test set NEVER touched during tuning — validation only.
    2. Fixed seed=42 everywhere.
    3. No invented numbers — all metrics from real executed runs.
    4. Member 1's source files (src/model.py, src/sensor_fusion.py) not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set matplotlib config dir before any pyplot import
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_har_tune"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.train import train_model  # all training logic (data loading, model, optimiser) lives there

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hyperparameter_tuning")

# ---------------------------------------------------------------------------
# Search Space  (3 × 3 × 3 × 3 × 2 = 162 configs per model)
# ---------------------------------------------------------------------------
LR_VALUES: List[float]       = [0.001, 0.0005, 0.0001]
BATCH_SIZES: List[int]       = [32, 64, 128]
DROPOUT_VALUES: List[float]  = [0.2, 0.3, 0.5]
MAX_EPOCHS_VALUES: List[int] = [20, 30, 50]
OPTIMIZERS: List[str]        = ["adam", "sgd"]

FULL_GRID: List[Dict[str, Any]] = [
    {"lr": lr, "batch_size": bs, "dropout": do, "max_epochs": ep, "optimizer": opt}
    for lr, bs, do, ep, opt in product(
        LR_VALUES, BATCH_SIZES, DROPOUT_VALUES, MAX_EPOCHS_VALUES, OPTIMIZERS
    )
]

# CSV columns — exact order required
CSV_COLUMNS: List[str] = [
    "model", "stage", "lr", "batch_size", "dropout", "optimizer",
    "max_epochs", "epochs_run", "best_epoch", "best_val_loss", "best_val_acc",
    "final_train_loss", "final_train_acc", "train_time_s", "seed", "status",
]

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR  = PROJECT_ROOT / "models"


# ===========================================================================
# Crash-Safe Duplicate Detection  (O(1) set-based)
# ===========================================================================

def _make_key(model: str, cfg: Dict[str, Any], seed: int) -> Tuple:
    """Hashable key for one (model, config, seed) triple."""
    return (
        str(model),
        float(cfg["lr"]),
        int(cfg["batch_size"]),
        float(cfg["dropout"]),
        str(cfg["optimizer"]),
        int(cfg["max_epochs"]),
        int(seed),
    )


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    """Load existing results CSV; return empty DataFrame with correct columns if absent."""
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} existing result(s) from {csv_path.name}")
            return df
        except Exception as exc:
            logger.warning(f"Could not read {csv_path}: {exc} — starting fresh.")
    return pd.DataFrame(columns=CSV_COLUMNS)


def build_done_keys(df: pd.DataFrame, seed: int) -> set:
    """
    Build a mutable set of _make_key tuples from an existing DataFrame.
    Used for O(1) in-session duplicate detection; the set is mutated after
    each successful run so we never double-run configs within one session.
    """
    if df.empty:
        return set()
    keys: set = set()
    for _, row in df.iterrows():
        try:
            keys.add((
                str(row["model"]),
                float(row["lr"]),
                int(row["batch_size"]),
                float(row["dropout"]),
                str(row["optimizer"]),
                int(row["max_epochs"]),
                int(row["seed"]),
            ))
        except (ValueError, KeyError):
            pass  # malformed row — ignore
    return keys


def write_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one result row to the CSV immediately (crash-safe)."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


# ===========================================================================
# Stratified Random Sampling
# ===========================================================================

def stratified_random_sample(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Sample N configs from the full 162-config grid with a fixed seed, ensuring
    every unique value of every hyperparameter dimension appears at least once.

    Strategy (Latin-hypercube inspired round-robin):
        1. For each (dimension, value) pair, pick one grid config containing
           that value from configs not yet chosen.  This covers all 14 unique
           dimension-values (3+3+3+3+2) in 14 "must-have" slots.
        2. Fill remaining N-14 slots by sampling without replacement from the
           un-chosen grid configs, shuffled with the fixed seed.
        3. If N < 14, fall back to pure random sampling (coverage not guaranteed).
        4. Final list is shuffled so the order is random, not dimension-grouped.

    Args:
        n   : Number of configs to sample.
        seed: RNG seed for reproducibility (default 42).

    Returns:
        List of config dicts of length min(n, 162).
    """
    rng = random.Random(seed)
    dimensions: Dict[str, List[Any]] = {
        "lr":         LR_VALUES,
        "batch_size": BATCH_SIZES,
        "dropout":    DROPOUT_VALUES,
        "max_epochs": MAX_EPOCHS_VALUES,
        "optimizer":  OPTIMIZERS,
    }

    def _cfg_tuple(c: Dict) -> Tuple:
        return (c["lr"], c["batch_size"], c["dropout"], c["max_epochs"], c["optimizer"])

    grid_copy = list(FULL_GRID)
    rng.shuffle(grid_copy)

    chosen: List[Dict[str, Any]] = []
    chosen_tuples: set = set()

    # Must-have pass: cover each (dim, val)
    for dim, vals in dimensions.items():
        for val in vals:
            if len(chosen) >= n:
                break
            candidates = [
                c for c in grid_copy
                if c[dim] == val and _cfg_tuple(c) not in chosen_tuples
            ]
            if not candidates:
                # All configs with this value were already chosen
                candidates = [c for c in FULL_GRID if c[dim] == val]
            pick = rng.choice(candidates)
            if _cfg_tuple(pick) not in chosen_tuples:
                chosen.append(pick)
                chosen_tuples.add(_cfg_tuple(pick))

    # Fill remaining slots with random unchosen configs
    remaining = [c for c in grid_copy if _cfg_tuple(c) not in chosen_tuples]
    rng.shuffle(remaining)
    slots_left = n - len(chosen)
    chosen.extend(remaining[:slots_left])

    # Shuffle so order is not dimension-grouped
    rng.shuffle(chosen)
    return chosen[:n]


# ===========================================================================
# Config Schedules by Mode
# ===========================================================================

def get_stage_a_configs() -> List[Dict[str, Any]]:
    """
    Stage A: 18 configs — full grid over lr × batch_size × optimizer,
    with dropout=0.3 and max_epochs=30 fixed.
    """
    return [
        {"lr": lr, "batch_size": bs, "dropout": 0.3, "max_epochs": 30, "optimizer": opt}
        for lr, bs, opt in product(LR_VALUES, BATCH_SIZES, OPTIMIZERS)
    ]


def select_top_k(df: pd.DataFrame, model: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    From a results DataFrame, select the top-k successful configs for 'model'
    by best_val_acc (descending), tie-broken by best_val_loss (ascending).
    """
    mdf = df[(df["model"] == model) & (df["status"] == "ok")].copy()
    if mdf.empty:
        return []
    mdf = mdf.sort_values(["best_val_acc", "best_val_loss"], ascending=[False, True])
    top = mdf.head(k)
    return [
        {
            "lr":         float(row["lr"]),
            "batch_size": int(row["batch_size"]),
            "optimizer":  str(row["optimizer"]),
            # dropout and max_epochs are swept in Stage B, so store Stage A values only
            # as the base; Stage B overrides them in the product loop below.
            "dropout":    float(row["dropout"]),
            "max_epochs": int(row["max_epochs"]),
        }
        for _, row in top.iterrows()
    ]


def get_stage_b_configs(stage_a_top3: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage B: for each of the top-3 Stage A base configs (lr, batch_size, optimizer),
    sweep dropout × max_epochs (3 × 3 = 9 combos each → 27 runs per model).
    """
    cfgs = []
    for base in stage_a_top3:
        for do, ep in product(DROPOUT_VALUES, MAX_EPOCHS_VALUES):
            cfgs.append({
                "lr":         base["lr"],
                "batch_size": base["batch_size"],
                "optimizer":  base["optimizer"],
                "dropout":    do,
                "max_epochs": ep,
            })
    return cfgs


# ===========================================================================
# Single-Run Executor
# ===========================================================================

def run_one(
    model_name: str,
    cfg: Dict[str, Any],
    stage: str,
    seed: int,
    csv_path: Path,
    models_dir: Path,
    results_dir: Path,
    done_keys: set,
    smoke: bool = False,
    smoke_max_epochs: int = 5,
) -> Dict[str, Any]:
    """
    Execute one hyperparameter configuration, write the row to CSV immediately.

    Args:
        model_name      : 'baseline' or 'adaptive'
        cfg             : dict with keys lr, batch_size, dropout, max_epochs, optimizer
        stage           : label written to CSV ('A', 'B', 'full', 'random')
        seed            : fixed RNG seed
        csv_path        : crash-safe output CSV
        models_dir      : directory for model checkpoints (smoke → temp dir)
        results_dir     : directory for run artifacts
        done_keys       : mutable set of _make_key tuples; updated after each run
                          so in-session duplicates are detected in O(1)
        smoke           : if True, cap epochs at smoke_max_epochs, use 256 samples
        smoke_max_epochs: epoch cap in smoke mode (default 5)

    Returns:
        Row dict (also written to CSV), or {} if config was skipped.
    """
    key = _make_key(model_name, cfg, seed)
    if key in done_keys:
        logger.info(
            f"[SKIP] {model_name} | stage={stage} | "
            f"lr={cfg['lr']} bs={cfg['batch_size']} do={cfg['dropout']} "
            f"ep={cfg['max_epochs']} opt={cfg['optimizer']} (already done)"
        )
        return {}

    effective_max_epochs = min(cfg["max_epochs"], smoke_max_epochs) if smoke else cfg["max_epochs"]

    logger.info(
        f"[RUN] {model_name} | stage={stage} | "
        f"lr={cfg['lr']} bs={cfg['batch_size']} do={cfg['dropout']} "
        f"ep={effective_max_epochs} opt={cfg['optimizer']}"
    )

    t0 = time.time()
    status = "ok"
    result: Dict[str, Any] = {}

    try:
        result = train_model(
            model_name=model_name,
            train_condition="normal",
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            dropout=cfg["dropout"],
            max_epochs=effective_max_epochs,
            optimizer_name=cfg["optimizer"],
            seed=seed,
            patience=7,
            use_scheduler=False,
            grad_clip=None,
            output_tag=f"tune_{stage}",
            results_dir=results_dir,
            models_dir=models_dir,
            smoke_test=smoke,
            smoke_samples=256,
        )
    except Exception as exc:
        logger.error(f"[ERROR] {model_name} run failed: {exc}", exc_info=True)
        status = f"error:{type(exc).__name__}"

    elapsed = time.time() - t0

    # Extract per-epoch history to populate epochs_run, final_train_loss, final_train_acc
    hist_df: Optional[pd.DataFrame] = result.get("history_df")
    if hist_df is not None and not hist_df.empty:
        epochs_run       = int(len(hist_df))
        final_train_loss = float(hist_df.iloc[-1]["train_loss"])
        final_train_acc  = float(hist_df.iloc[-1]["train_acc"])
    else:
        epochs_run       = 0
        final_train_loss = float("nan")
        final_train_acc  = float("nan")

    row: Dict[str, Any] = {
        "model":            model_name,
        "stage":            stage,
        "lr":               cfg["lr"],
        "batch_size":       cfg["batch_size"],
        "dropout":          cfg["dropout"],
        "optimizer":        cfg["optimizer"],
        "max_epochs":       cfg["max_epochs"],
        "epochs_run":       epochs_run,
        "best_epoch":       result.get("best_epoch", -1),
        "best_val_loss":    result.get("best_val_loss", float("nan")),
        "best_val_acc":     result.get("best_val_acc", float("nan")),
        "final_train_loss": final_train_loss,
        "final_train_acc":  final_train_acc,
        "train_time_s":     round(elapsed, 1),
        "seed":             seed,
        "status":           status,
    }

    # Write row immediately (crash-safe)
    write_row(csv_path, row)
    # Update in-memory key set so the same config isn't retried later this session
    done_keys.add(key)

    logger.info(
        f"[DONE] {model_name} | stage={stage} | "
        f"best_val_acc={row['best_val_acc']:.4f} best_val_loss={row['best_val_loss']:.4f} "
        f"epochs_run={row['epochs_run']} time={row['train_time_s']:.0f}s status={status}"
    )
    return row


# ===========================================================================
# Post-Processing: Best Config Selection, Summary Plot, Ranked Table
# ===========================================================================

def select_best_per_model(df: pd.DataFrame, models: List[str]) -> Dict[str, Any]:
    """
    Choose the best config per model by best_val_acc (tie-break: lower best_val_loss).
    All numbers come from the real CSV — nothing is hardcoded.

    Returns:
        Dict mapping model name → config dict (with metrics).
    """
    best: Dict[str, Any] = {}
    for model in models:
        mdf = df[(df["model"] == model) & (df["status"] == "ok")].copy()
        if mdf.empty:
            logger.warning(f"No successful runs for '{model}' — skipping best-config selection.")
            continue
        mdf = mdf.sort_values(["best_val_acc", "best_val_loss"], ascending=[False, True])
        r = mdf.iloc[0]
        best[model] = {
            "model":         model,
            "lr":            float(r["lr"]),
            "batch_size":    int(r["batch_size"]),
            "dropout":       float(r["dropout"]),
            "optimizer":     str(r["optimizer"]),
            "max_epochs":    int(r["max_epochs"]),
            "best_val_acc":  float(r["best_val_acc"]),
            "best_val_loss": float(r["best_val_loss"]),
            "best_epoch":    int(r["best_epoch"]),
            "epochs_run":    int(r["epochs_run"]),
            "train_time_s":  float(r["train_time_s"]),
            "stage":         str(r["stage"]),
            "seed":          int(r["seed"]),
        }
    return best


def print_top5_table(df: pd.DataFrame, models: List[str]) -> None:
    """Print a ranked top-5 table per model to stdout."""
    display_cols = [
        "model", "stage", "lr", "batch_size", "dropout", "optimizer",
        "max_epochs", "epochs_run", "best_epoch", "best_val_acc", "best_val_loss",
        "train_time_s",
    ]
    for model in models:
        mdf = df[(df["model"] == model) & (df["status"] == "ok")].copy()
        if mdf.empty:
            print(f"\n[{model.upper()}] No successful runs.\n")
            continue
        mdf = mdf.sort_values(["best_val_acc", "best_val_loss"], ascending=[False, True])
        top5 = mdf.head(5)[display_cols].reset_index(drop=True)
        top5.index += 1  # 1-based rank
        top5["best_val_acc"]  = top5["best_val_acc"].map(lambda x: f"{float(x):.4f}")
        top5["best_val_loss"] = top5["best_val_loss"].map(lambda x: f"{float(x):.4f}")
        top5["lr"]            = top5["lr"].map(lambda x: f"{float(x):.5f}")
        print(f"\n{'='*80}")
        print(f"  TOP-5 CONFIGS — {model.upper()}")
        print(f"{'='*80}")
        print(top5.to_string())
        print()


def save_summary_plot(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a heatmap of best val_acc by lr × batch_size, one panel per optimizer.
    All values in the plot are read from the real results CSV.
    """
    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        logger.warning("No successful runs — skipping summary plot.")
        return

    optimizers = sorted(ok_df["optimizer"].unique().tolist())
    n_opts = len(optimizers)

    fig, axes = plt.subplots(1, n_opts, figsize=(6 * n_opts, 5), squeeze=False)
    fig.suptitle(
        "Hyperparameter Tuning — Best Val Acc by LR × Batch Size",
        fontsize=14, y=1.02,
    )

    vmin = max(0.0, float(ok_df["best_val_acc"].min()) - 0.05)
    vmax = float(ok_df["best_val_acc"].max()) + 0.01

    for col_idx, opt in enumerate(optimizers):
        ax = axes[0][col_idx]
        sub = ok_df[ok_df["optimizer"] == opt]

        # Pivot: for each (lr, batch_size) take the max val acc over all other dims
        pivot = sub.pivot_table(
            index="lr",
            columns="batch_size",
            values="best_val_acc",
            aggfunc="max",
        )
        pivot = pivot.sort_index(ascending=False)  # highest lr at top

        try:
            import seaborn as sns
            sns.heatmap(
                pivot, ax=ax,
                annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=vmin, vmax=vmax,
                linewidths=0.5,
                cbar_kws={"label": "Best Val Acc"},
            )
        except ImportError:
            im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns.tolist())
            ax.set_yticklabels([f"{v:.4f}" for v in pivot.index.tolist()])
            plt.colorbar(im, ax=ax, label="Best Val Acc")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

        ax.set_title(f"Optimizer: {opt.upper()}", fontsize=12)
        ax.set_xlabel("Batch Size", fontsize=10)
        ax.set_ylabel("Learning Rate", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary plot saved → {out_path}")


# ===========================================================================
# Main Orchestrator
# ===========================================================================

def run_tuning(
    models: List[str],
    mode: str,
    max_runs: Optional[int],
    time_budget_minutes: Optional[float],
    seed: int,
    csv_path: Path,
    results_dir: Path,
    models_dir: Path,
    smoke: bool,
    smoke_max_epochs: int = 5,
) -> None:
    """
    Main tuning loop. Dispatches to staged / full / random config schedules,
    enforces max_runs and time_budget_minutes, writes CSV crash-safely.

    Duplicate detection uses a mutable set of run keys (populated from any
    pre-existing CSV on startup), so the same config is never run twice — even
    across sessions — without re-reading the full CSV on every iteration.

    Args:
        models              : list of model names to tune ('baseline', 'adaptive')
        mode                : 'staged', 'full', or 'random'
        max_runs            : stop after this many configs per model (None = unlimited)
        time_budget_minutes : global wall-clock budget; stops before a new run if
                              remaining budget < estimated 1 epoch of training time
        seed                : fixed RNG seed
        csv_path            : output CSV path
        results_dir         : results directory
        models_dir          : model checkpoint directory
        smoke               : if True, use smoke mode (tiny data, capped epochs)
        smoke_max_epochs    : epoch cap in smoke mode
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    existing_df = load_existing_results(csv_path)
    # Build in-memory set for O(1) dup detection; mutated inside run_one
    done_keys = build_done_keys(existing_df, seed)
    logger.info(f"Pre-existing done configs: {len(done_keys)}")

    global_start = time.time()

    def _time_remaining_s() -> Optional[float]:
        if time_budget_minutes is None:
            return None
        return time_budget_minutes * 60.0 - (time.time() - global_start)

    def _budget_ok(estimated_epoch_s: float = 15.0) -> bool:
        """Return True if at least one epoch's worth of budget remains."""
        rem = _time_remaining_s()
        return rem is None or rem > estimated_epoch_s

    # Common kwargs for run_one
    run_kwargs = dict(
        seed=seed,
        csv_path=csv_path,
        models_dir=models_dir,
        results_dir=results_dir,
        done_keys=done_keys,
        smoke=smoke,
        smoke_max_epochs=smoke_max_epochs,
    )

    for model_name in models:
        logger.info(
            f"\n{'='*60}\n  Tuning: {model_name.upper()} | mode={mode} | "
            f"max_runs={max_runs}\n{'='*60}"
        )
        runs_this_model = 0

        # ------------------------------------------------------------------
        if mode == "staged":
            # Stage A — 18 configs
            stage_a_cfgs = get_stage_a_configs()
            logger.info(f"Stage A: {len(stage_a_cfgs)} configs for {model_name}")

            for cfg in stage_a_cfgs:
                if max_runs is not None and runs_this_model >= max_runs:
                    logger.info(f"[{model_name}] --max-runs {max_runs} reached in Stage A.")
                    break
                if not _budget_ok():
                    logger.info(f"[{model_name}] Time budget exhausted in Stage A.")
                    break
                row = run_one(model_name, cfg, stage="A", **run_kwargs)
                if row:
                    runs_this_model += 1

            # Reload CSV from disk for Stage B selection (includes pre-existing Stage A rows)
            stage_a_df = load_existing_results(csv_path)
            stage_a_df = stage_a_df[
                (stage_a_df["model"] == model_name) & (stage_a_df["stage"] == "A")
            ]
            top3 = select_top_k(stage_a_df, model_name, k=3)
            if not top3:
                logger.warning(f"[{model_name}] No valid Stage A results — skipping Stage B.")
            else:
                stage_b_cfgs = get_stage_b_configs(top3)
                logger.info(f"Stage B: {len(stage_b_cfgs)} configs for {model_name}")
                for cfg in stage_b_cfgs:
                    if max_runs is not None and runs_this_model >= max_runs:
                        logger.info(f"[{model_name}] --max-runs {max_runs} reached in Stage B.")
                        break
                    if not _budget_ok():
                        logger.info(f"[{model_name}] Time budget exhausted in Stage B.")
                        break
                    row = run_one(model_name, cfg, stage="B", **run_kwargs)
                    if row:
                        runs_this_model += 1

        # ------------------------------------------------------------------
        elif mode == "full":
            logger.info(f"Full grid: {len(FULL_GRID)} configs for {model_name}")
            for cfg in FULL_GRID:
                if max_runs is not None and runs_this_model >= max_runs:
                    logger.info(f"[{model_name}] --max-runs {max_runs} reached.")
                    break
                if not _budget_ok():
                    logger.info(f"[{model_name}] Time budget exhausted.")
                    break
                row = run_one(model_name, cfg, stage="full", **run_kwargs)
                if row:
                    runs_this_model += 1

        # ------------------------------------------------------------------
        else:  # random (default)
            n_sample = max_runs if max_runs is not None else 20
            cfgs = stratified_random_sample(n_sample, seed=seed)
            logger.info(
                f"Random stratified: {len(cfgs)} configs sampled for {model_name} (seed={seed})"
            )
            for cfg in cfgs:
                if not _budget_ok():
                    logger.info(f"[{model_name}] Time budget exhausted.")
                    break
                row = run_one(model_name, cfg, stage="random", **run_kwargs)
                if row:
                    runs_this_model += 1

        logger.info(f"[{model_name}] Configs executed this session: {runs_this_model}")

    # --- Post-Processing ---
    final_df = load_existing_results(csv_path)
    if final_df.empty or (final_df["status"] == "ok").sum() == 0:
        logger.warning("No successful runs in CSV — skipping post-processing.")
        return

    # Best configs JSON (all numbers from real CSV, never hardcoded)
    best = select_best_per_model(final_df, models)
    best_json_path = results_dir / "best_hyperparameters.json"
    with open(best_json_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"Best hyperparameters → {best_json_path}")

    # Ranked table
    print_top5_table(final_df, models)

    # Summary plot
    summary_plot_path = results_dir / "hyperparameter_summary.png"
    save_summary_plot(final_df, summary_plot_path)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning for HAR sensor fusion models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["staged", "full", "random"],
        default="random",
        help="Search strategy: staged (A+B), full (all 162), random (stratified sample).",
    )
    p.add_argument(
        "--models",
        default="both",
        help=(
            "Comma-separated model names or 'both'. "
            "E.g. 'baseline,adaptive' or 'baseline'."
        ),
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Maximum configs to run per model. "
            "In 'random' mode defaults to 20 when unset."
        ),
    )
    p.add_argument(
        "--time-budget-minutes",
        type=float,
        default=None,
        help=(
            "Global wall-clock time budget (minutes). Tuning stops before a "
            "new run when remaining budget < ~1 estimated epoch of training."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global RNG seed for reproducibility.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Smoke-test mode: tiny data subset, epochs capped at --smoke-epochs, "
            "output written to a temporary directory."
        ),
    )
    p.add_argument(
        "--smoke-epochs", type=int, default=5,
        help="Epoch cap used in smoke mode.",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Override output directory for CSV and plots (default: results/).",
    )
    p.add_argument(
        "--models-dir", type=Path, default=None,
        help="Override output directory for model checkpoints (default: models/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve model list
    raw = args.models.lower().strip()
    if raw == "both":
        models = ["baseline", "adaptive"]
    else:
        models = [m.strip() for m in raw.split(",") if m.strip()]
    valid = {"baseline", "adaptive"}
    for m in models:
        if m not in valid:
            raise ValueError(f"Unknown model '{m}'. Valid: {valid}")

    # Resolve directories (smoke → temp dir)
    if args.smoke:
        smoke_tmp = Path(tempfile.mkdtemp(prefix="har_tune_smoke_"))
        results_dir = smoke_tmp / "results"
        models_dir  = smoke_tmp / "models"
        logger.info(f"[SMOKE] Temp output directory: {smoke_tmp}")
    else:
        results_dir = args.results_dir or RESULTS_DIR
        models_dir  = args.models_dir  or MODELS_DIR

    csv_path = results_dir / "hyperparameter_results.csv"

    # In random mode with no --max-runs, default to 20
    max_runs = args.max_runs
    if args.mode == "random" and max_runs is None:
        max_runs = 20
        logger.info(f"Random mode: defaulting --max-runs to {max_runs} per model.")

    logger.info(
        f"Hyperparameter Tuning | mode={args.mode} | models={models} | "
        f"max_runs={max_runs} | budget={args.time_budget_minutes} min | "
        f"smoke={args.smoke} | seed={args.seed}"
    )

    run_tuning(
        models=models,
        mode=args.mode,
        max_runs=max_runs,
        time_budget_minutes=args.time_budget_minutes,
        seed=args.seed,
        csv_path=csv_path,
        results_dir=results_dir,
        models_dir=models_dir,
        smoke=args.smoke,
        smoke_max_epochs=args.smoke_epochs,
    )

    logger.info("Hyperparameter tuning complete.")


if __name__ == "__main__":
    main()
