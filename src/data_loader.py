#!.venv/Scripts/python.exe
"""
src/data_loader.py

Phase 3/4 Data Pipeline for Human Activity Recognition (HAR) using MotionSense.
Owned by Member 2.

Key Responsibilities:
    1. CSV Ingestion & Header Auto-Detection:
       Loads motionsense_output/motionsense_preprocessed.csv, auto-detects column names,
       validates raw sensor features, and sorts samples chronologically per trial.
    2. Strict Subject-Wise 70/15/15 Split:
       Splits 24 subjects into 17 Train (~70.83%), 3 Val (~12.50%), and 4 Test (~16.67%).
       Persists subject split to results/subject_split.json for identical reuse by Member 3.
    3. Leak-Free Scaling:
       Fits a new StandardScaler exclusively on raw sensor readings of the training subjects.
       Saves scaler to models/train_only_scaler.pkl. Transforms Val and Test without leaking
       distribution statistics.
    4. Boundary-Guarded Temporal Windowing:
       Generates windows of length 100 (stride 50) strictly within trial boundaries.
       Trailing partial windows are discarded. Keeps per-window metadata.
    5. PyTorch Dataset & DataLoader Interface:
       Supplies MotionSenseDataset returning (x, y) with x shaped (12, 100) matching model input.
       Provides get_dataloaders() with seeded train shuffling and un-shuffled val/test.
       Provides optional compute_class_weights().
    6. Compressed Caching:
       Caches processed window arrays to motionsense_output/windows_w{W}_s{S}_seed{SEED}.npz.
    7. Summary Reporting:
       Exports comprehensive dataset metrics to results/data_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path to support execution as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset

from src.model import NUM_ACTIVITIES, NUM_SENSOR_FEATURES, SENSOR_COLUMNS

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_loader")

# ============================================================================
# Project Path Constants
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "motionsense_output"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFAULT_CSV_PATH = OUTPUT_DIR / "motionsense_preprocessed.csv"
DEFAULT_SPLIT_PATH = RESULTS_DIR / "subject_split.json"
DEFAULT_SCALER_PATH = MODELS_DIR / "train_only_scaler.pkl"
DEFAULT_LABEL_MAPPING_PATH = OUTPUT_DIR / "activity_label_mapping.json"
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "data_summary.json"


# ============================================================================
# Column Auto-Detection & Verification
# ============================================================================
def detect_columns(columns: List[str]) -> Dict[str, str]:
    """
    Auto-detects semantic column names from the CSV header.

    Args:
        columns: List of column names present in the preprocessed CSV.

    Returns:
        Mapping dictionary with canonical keys:
        'subject_id', 'activity', 'trial', 'source_file', 'sample_index', 'activity_encoded'.

    Raises:
        KeyError: If any required column role cannot be identified.
    """
    col_lookup = {col.lower().strip(): col for col in columns}
    mapping: Dict[str, str] = {}

    candidates: Dict[str, List[str]] = {
        "subject_id": ["subject_id", "subject", "participant_id", "participant", "sub_id", "code"],
        "activity": ["activity", "activity_name", "label_name", "target_name"],
        "trial": ["trial", "trial_id", "trial_num", "run"],
        "source_file": ["source_file", "file", "filename", "filepath"],
        "sample_index": ["sample_index", "index", "time_index", "sample", "unnamed: 0"],
        "activity_encoded": ["activity_encoded", "encoded_activity", "label_encoded", "target_encoded"],
    }

    for role, candidate_names in candidates.items():
        for name in candidate_names:
            if name in col_lookup:
                mapping[role] = col_lookup[name]
                break
        if role not in mapping:
            raise KeyError(
                f"Auto-detection failed: unable to resolve required column '{role}'. "
                f"Available CSV columns: {columns}"
            )

    return mapping


def validate_raw_sensor_columns(columns: List[str]) -> None:
    """Ensures all 12 canonical raw sensor feature columns are present."""
    missing = [c for c in SENSOR_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"Missing required raw sensor feature columns: {missing}. "
            f"Expected 12 features in exact order: {SENSOR_COLUMNS}"
        )


# ============================================================================
# Subject-Wise Splitting (70/15/15)
# ============================================================================
def get_subject_split(
    subject_ids: List[int],
    seed: int = 42,
    split_path: Optional[Path] = None,
) -> Dict[str, List[int]]:
    """
    Computes or reloads a reproducible, disjoint subject-wise 70/15/15 split.

    Rounding Justification (Total Subjects N = 24):
        - Train target: 70% of 24 = 16.8 -> rounded to 17 participants (~70.83%).
        - Remaining: 24 - 17 = 7 participants (29.17%).
        - Val target:  15% of 24 = 3.6  -> allocated 3 participants (~12.50%).
        - Test target: 15% of 24 = 3.6  -> allocated 4 participants (~16.67%).
        Allocating 4 subjects to Test rather than Val provides a statistically
        stronger held-out evaluation benchmark for final robustness stress-testing
        (Member 3), while maintaining ~71% of data for training.

    Args:
        subject_ids: List of unique participant IDs (1-24).
        seed: Random seed for deterministic permutation.
        split_path: Optional destination to save/load JSON record.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to subject ID lists.
    """
    target_path = split_path or DEFAULT_SPLIT_PATH

    # Reload cached split if it exists so all members use identical subjects
    if target_path.exists():
        logger.info(f"Loading existing subject split from: {target_path}")
        return load_split(target_path)

    unique_subjects = sorted(list(set(subject_ids)))
    if len(unique_subjects) != 24:
        logger.warning(
            f"Expected 24 unique subjects, but received {len(unique_subjects)}. "
            f"Adjusting proportional splits accordingly."
        )

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_subjects).tolist()

    n_total = len(shuffled)
    n_train = 17 if n_total == 24 else int(round(n_total * 0.70))
    n_val = 3 if n_total == 24 else int(round(n_total * 0.15))
    n_test = n_total - n_train - n_val

    train_subjects = sorted(shuffled[:n_train])
    val_subjects = sorted(shuffled[n_train : n_train + n_val])
    test_subjects = sorted(shuffled[n_train + n_val :])

    # Assert mutual disjointness
    set_train = set(train_subjects)
    set_val = set(val_subjects)
    set_test = set(test_subjects)
    assert set_train.isdisjoint(set_val), "Train and Val subject sets overlap!"
    assert set_train.isdisjoint(set_test), "Train and Test subject sets overlap!"
    assert set_val.isdisjoint(set_test), "Val and Test subject sets overlap!"
    assert len(set_train | set_val | set_test) == n_total, "Subject partition is non-exhaustive!"

    split_dict = {
        "seed": seed,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "counts": {
            "train": len(train_subjects),
            "val": len(val_subjects),
            "test": len(test_subjects),
            "total": n_total,
        },
        "proportions": {
            "train": round(len(train_subjects) / n_total, 4),
            "val": round(len(val_subjects) / n_total, 4),
            "test": round(len(test_subjects) / n_total, 4),
        },
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=4)
    logger.info(f"Saved new subject split to: {target_path}")

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


def load_split(split_path: Optional[Path] = None) -> Dict[str, List[int]]:
    """Loads a previously saved subject partition JSON file."""
    path = split_path or DEFAULT_SPLIT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "train": data["train_subjects"],
        "val": data["val_subjects"],
        "test": data["test_subjects"],
    }


# ============================================================================
# Leak-Free Sensor Scaling
# ============================================================================
def fit_and_apply_scaler(
    df: pd.DataFrame,
    train_subjects: List[int],
    scaler_save_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fits StandardScaler exclusively on training subject data, transforming all rows.

    Data Leakage Prevention Justification:
        Phase 2's sensor_scaler.pkl was fitted globally across all 24 subjects,
        meaning test participant motion dynamics leaked into the mean and variance
        estimates. To ensure rigorous, zero-leakage evaluation in Phase 3/4,
        we fit a fresh StandardScaler on RAW sensor columns from training subjects
        only, preserving the test set as completely unseen out-of-distribution data.

    Args:
        df: Full MotionSense DataFrame containing raw SENSOR_COLUMNS.
        train_subjects: List of subject IDs allocated to the training split.
        scaler_save_path: Destination path for models/train_only_scaler.pkl.

    Returns:
        A tuple of (transformed_dataframe, fitted_StandardScaler).
    """
    logger.info("Fitting leak-free StandardScaler exclusively on training-subject rows...")
    train_mask = df["subject_id"].isin(train_subjects)
    train_raw = df.loc[train_mask, SENSOR_COLUMNS].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(train_raw)

    save_path = scaler_save_path or DEFAULT_SCALER_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    logger.info(f"Saved training-only scaler to: {save_path}")

    # Transform full dataset using training statistics
    scaled_values = scaler.transform(df[SENSOR_COLUMNS].to_numpy(dtype=np.float32))
    df_scaled = df.copy()
    df_scaled[SENSOR_COLUMNS] = scaled_values

    return df_scaled, scaler


# ============================================================================
# Boundary-Guarded Windowing
# ============================================================================
def extract_windows_from_trials(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    window_size: int = 100,
    stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Segments continuous time-series trials into fixed-length windows.

    Guarantees:
        - Windows NEVER span trial or participant boundaries.
        - Processing is grouped by (subject_id, trial, activity_encoded, source_file).
        - Trailing partial windows (< window_size) are strictly dropped.
        - Each window maintains associated metadata (subject, trial, activity, indices).

    Args:
        df: DataFrame containing sorted sensor readings and trial identifiers.
        col_map: Detected column mapping dictionary.
        window_size: Number of consecutive time-steps per window (default 100).
        stride: Step size between consecutive window starts (default 50).

    Returns:
        X: float32 numpy array of shape (N_windows, window_size, 12).
        y: int64 numpy array of shape (N_windows,).
        metadata: List of dictionaries detailing source file, subject, trial, and sample range.
    """
    sub_col = col_map["subject_id"]
    trial_col = col_map["trial"]
    act_enc_col = col_map["activity_encoded"]
    act_str_col = col_map["activity"]
    src_col = col_map["source_file"]
    idx_col = col_map["sample_index"]

    # Group trials deterministically
    group_cols = [sub_col, trial_col, act_enc_col, src_col]
    groups = df.groupby(group_cols, sort=False)

    windows_list: List[np.ndarray] = []
    labels_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    for (sub_id, trial_num, act_code, src_file), group_df in groups:
        # Guarantee strict chronological ordering by sample index
        group_df_sorted = group_df.sort_values(by=idx_col)
        n_samples = len(group_df_sorted)

        if n_samples < window_size:
            # Drop trial if insufficient data for even one full window
            continue

        sensor_data = group_df_sorted[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
        sample_indices = group_df_sorted[idx_col].to_numpy()
        activity_name = str(group_df_sorted[act_str_col].iloc[0])
        label = int(act_code)

        # Sliding window extraction
        for start_idx in range(0, n_samples - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = sensor_data[start_idx:end_idx]

            windows_list.append(window)
            labels_list.append(label)
            meta_list.append(
                {
                    "subject_id": int(sub_id),
                    "trial": int(trial_num),
                    "activity": activity_name,
                    "activity_encoded": label,
                    "source_file": str(src_file),
                    "start_sample_index": int(sample_indices[start_idx]),
                    "end_sample_index": int(sample_indices[end_idx - 1]),
                }
            )

    if not windows_list:
        X = np.empty((0, window_size, NUM_SENSOR_FEATURES), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.stack(windows_list, axis=0).astype(np.float32)
        y = np.array(labels_list, dtype=np.int64)

    return X, y, meta_list


# ============================================================================
# PyTorch Dataset
# ============================================================================
class MotionSenseDataset(Dataset):
    """
    PyTorch Dataset serving MotionSense multi-sensor temporal windows.

    Data Layout:
        Returns items formatted for BaselineCNN and AdaptiveSensorFusion:
            x: Tensor of shape (12, sequence_length), float32.
            y: Tensor scalar, int64.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        return_metadata: bool = False,
    ) -> None:
        """
        Args:
            X: Input windows array of shape (N, sequence_length, 12).
            y: Target class array of shape (N,).
            metadata: Optional list of window metadata records.
            return_metadata: If True, __getitem__ returns (x, y, meta).
        """
        super().__init__()
        if X.ndim != 3 or X.shape[2] != NUM_SENSOR_FEATURES:
            raise ValueError(
                f"Expected X with shape (N, sequence_length, {NUM_SENSOR_FEATURES}), but got {X.shape}"
            )
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} windows, y has {len(y)} labels.")

        # Pre-transpose (N, L, 12) -> (N, 12, L) so each __getitem__ requires no runtime transpose
        self.X = torch.from_numpy(np.transpose(X, (0, 2, 1))).float().contiguous()
        self.y = torch.from_numpy(y).long()
        self.metadata = metadata
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        if self.return_metadata and self.metadata is not None:
            return self.X[idx], self.y[idx], self.metadata[idx]
        return self.X[idx], self.y[idx]


# ============================================================================
# Class Weighting Utility
# ============================================================================
def compute_class_weights(
    y_train: Union[np.ndarray, torch.Tensor],
    num_classes: int = NUM_ACTIVITIES,
) -> torch.Tensor:
    """
    Computes balanced class weights for CrossEntropyLoss:
        w_c = total_samples / (num_classes * count_c)

    Optional tool for imbalanced training; not enabled by default.

    Args:
        y_train: Integer class labels of the training dataset.
        num_classes: Total number of discrete activity classes (default 6).

    Returns:
        torch.Tensor of shape (num_classes,) with float32 weights.
    """
    if isinstance(y_train, torch.Tensor):
        y_arr = y_train.cpu().numpy()
    else:
        y_arr = np.asarray(y_train)

    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_arr,
    )
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# End-to-End Data Pipeline & Caching
# ============================================================================
def prepare_windowed_data(
    csv_path: Optional[Path] = None,
    window_size: int = 100,
    stride: int = 50,
    seed: int = 42,
    force_rebuild: bool = False,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Loads preprocessed MotionSense records, scales without leakage, extracts windows,
    caches to disk, and records data summary metrics.

    Args:
        csv_path: Path to motionsense_preprocessed.csv.
        window_size: Window length in readings (default 100).
        stride: Stride step in readings (default 50).
        seed: Random seed for subject splitting (default 42).
        force_rebuild: If True, ignores existing cache file and recomputes.
        cache_dir: Directory for storing the compressed .npz cache.

    Returns:
        Dictionary containing X and y arrays and metadata for train, val, and test splits.
    """
    source_csv = csv_path or DEFAULT_CSV_PATH
    c_dir = cache_dir or OUTPUT_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"windows_w{window_size}_s{stride}_seed{seed}.npz"

    # 1. Load from cache if available
    if cache_file.exists() and not force_rebuild:
        logger.info(f"Loading cached windowed data from: {cache_file}")
        with np.load(cache_file, allow_pickle=True) as data:
            result = {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "meta_train": data["meta_train"].tolist(),
                "X_val": data["X_val"],
                "y_val": data["y_val"],
                "meta_val": data["meta_val"].tolist(),
                "X_test": data["X_test"],
                "y_test": data["y_test"],
                "meta_test": data["meta_test"].tolist(),
                "split": load_split(),
            }
        return result

    # 2. Build from raw preprocessed CSV
    if not source_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed MotionSense CSV not found at '{source_csv}'. "
            f"Run 'data scripts/run_pipeline.py' first."
        )

    logger.info(f"Reading preprocessed dataset from: {source_csv}")
    df = pd.read_csv(source_csv)

    # Validate header & auto-detect column semantics
    col_map = detect_columns(list(df.columns))
    validate_raw_sensor_columns(list(df.columns))

    # Assert no NaN or infinite values
    check_cols = SENSOR_COLUMNS + [col_map["activity_encoded"], col_map["subject_id"]]
    if df[check_cols].isna().any().any():
        raise ValueError("Dataset contains unexpected NaN values in essential sensor or label columns.")
    if np.isinf(df[SENSOR_COLUMNS].to_numpy()).any():
        raise ValueError("Dataset contains unexpected infinite values in raw sensor features.")

    # 3. Compute or load subject partition
    unique_subjects = df[col_map["subject_id"]].unique().tolist()
    split = get_subject_split(unique_subjects, seed=seed)

    # 4. Leak-Free Scaling: fit strictly on train subjects
    df_scaled, _ = fit_and_apply_scaler(df, split["train"])

    # 5. Extract temporal windows per split
    logger.info(f"Windowing data (window_size={window_size}, stride={stride})...")
    sub_col = col_map["subject_id"]

    df_train = df_scaled[df_scaled[sub_col].isin(split["train"])]
    df_val = df_scaled[df_scaled[sub_col].isin(split["val"])]
    df_test = df_scaled[df_scaled[sub_col].isin(split["test"])]

    X_train, y_train, meta_train = extract_windows_from_trials(df_train, col_map, window_size, stride)
    X_val, y_val, meta_val = extract_windows_from_trials(df_val, col_map, window_size, stride)
    X_test, y_test, meta_test = extract_windows_from_trials(df_test, col_map, window_size, stride)

    logger.info(
        f"Windowing complete: Train={len(y_train):,} | Val={len(y_val):,} | Test={len(y_test):,} "
        f"(Total={len(y_train) + len(y_val) + len(y_test):,})"
    )

    # 6. Save compressed cache
    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        meta_train=np.array(meta_train, dtype=object),
        X_val=X_val,
        y_val=y_val,
        meta_val=np.array(meta_val, dtype=object),
        X_test=X_test,
        y_test=y_test,
        meta_test=np.array(meta_test, dtype=object),
    )
    logger.info(f"Cached windowed dataset to: {cache_file}")

    # 7. Compute & Save Summary Metrics
    save_data_summary(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        split=split,
        window_size=window_size,
        stride=stride,
        seed=seed,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
        "split": split,
    }


def save_data_summary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    split: Dict[str, List[int]],
    window_size: int,
    stride: int,
    seed: int,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generates and persists dataset summary statistics to results/data_summary.json."""
    label_map_file = DEFAULT_LABEL_MAPPING_PATH
    if label_map_file.exists():
        with open(label_map_file, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {f"class_{i}": i for i in range(NUM_ACTIVITIES)}

    inv_label_map = {v: k for k, v in label_mapping.items()}

    def get_class_counts(y_arr: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(y_arr, return_counts=True)
        count_dict = {inv_label_map.get(int(u), f"class_{u}"): int(c) for u, c in zip(unique, counts)}
        # Ensure all classes exist in dictionary
        for name in label_mapping.keys():
            count_dict.setdefault(name, 0)
        return dict(sorted(count_dict.items()))

    summary = {
        "seed": seed,
        "window_size": window_size,
        "stride": stride,
        "overlap_percentage": round((1.0 - stride / window_size) * 100, 1),
        "num_sensor_features": NUM_SENSOR_FEATURES,
        "feature_order": SENSOR_COLUMNS,
        "input_shape_array": [window_size, NUM_SENSOR_FEATURES],
        "input_shape_model": [NUM_SENSOR_FEATURES, window_size],
        "participants_per_split": {
            "train": split["train"],
            "val": split["val"],
            "test": split["test"],
        },
        "participant_counts": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
            "total": len(split["train"]) + len(split["val"]) + len(split["test"]),
        },
        "total_windows_per_split": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "total": int(len(y_train) + len(y_val) + len(y_test)),
        },
        "class_counts_per_split": {
            "train": get_class_counts(y_train),
            "val": get_class_counts(y_val),
            "test": get_class_counts(y_test),
        },
    }

    target = summary_path or DEFAULT_SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved dataset summary to: {target}")

    return summary


# ============================================================================
# PyTorch DataLoader Factory
# ============================================================================
def get_dataloaders(
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    window_size: int = 100,
    stride: int = 50,
    data_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds PyTorch DataLoaders for train, validation, and test sets.

    Configuration:
        - Train loader shuffles using a seeded torch.Generator(seed).
        - Val and test loaders do NOT shuffle (shuffle=False).
        - pin_memory enabled automatically when CUDA GPU is detected.

    Args:
        batch_size: Mini-batch size (default 64).
        seed: Random seed for shuffling reproducibility.
        num_workers: Worker subprocesses for data loading (default 0).
        window_size: Consecutive sensor readings per window (default 100).
        stride: Temporal window step size (default 50).
        data_path: Optional custom CSV source file.
        cache_dir: Optional custom directory for .npz cache.
        force_rebuild: If True, forces re-computation of windows.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    csv_file = Path(data_path) if data_path else DEFAULT_CSV_PATH
    c_dir = Path(cache_dir) if cache_dir else OUTPUT_DIR

    data_dict = prepare_windowed_data(
        csv_path=csv_file,
        window_size=window_size,
        stride=stride,
        seed=seed,
        force_rebuild=force_rebuild,
        cache_dir=c_dir,
    )

    train_dataset = MotionSenseDataset(
        X=data_dict["X_train"],
        y=data_dict["y_train"],
        metadata=data_dict["meta_train"],
    )
    val_dataset = MotionSenseDataset(
        X=data_dict["X_val"],
        y=data_dict["y_val"],
        metadata=data_dict["meta_val"],
    )
    test_dataset = MotionSenseDataset(
        X=data_dict["X_test"],
        y=data_dict["y_test"],
        metadata=data_dict["meta_test"],
    )

    # Seeded generator for reproducible training batch order
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Self-Test Verification Routine
# ============================================================================
def run_self_test() -> None:
    """Comprehensive test assertions verifying all requirements."""
    print("=" * 78)
    print("  RUNNING src/data_loader.py COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 78)

    # 1. Build and verify windowed data
    window_size, stride, seed = 100, 50, 42
    data_dict = prepare_windowed_data(
        window_size=window_size,
        stride=stride,
        seed=seed,
        force_rebuild=False,
    )

    split = data_dict["split"]
    X_train, y_train, meta_train = data_dict["X_train"], data_dict["y_train"], data_dict["meta_train"]
    X_val, y_val, meta_val = data_dict["X_val"], data_dict["y_val"], data_dict["meta_val"]
    X_test, y_test, meta_test = data_dict["X_test"], data_dict["y_test"], data_dict["meta_test"]

    print("\n[Check 1] Subject Partition (70/15/15):")
    print(f"  Train ({len(split['train'])} subjects): {split['train']}")
    print(f"  Val   ({len(split['val'])} subjects): {split['val']}")
    print(f"  Test  ({len(split['test'])} subjects): {split['test']}")

    # Assert mutual disjointness
    set_train, set_val, set_test = set(split["train"]), set(split["val"]), set(split["test"])
    assert set_train.isdisjoint(set_val), "Assertion failed: Train and Val subjects overlap!"
    assert set_train.isdisjoint(set_test), "Assertion failed: Train and Test subjects overlap!"
    assert set_val.isdisjoint(set_test), "Assertion failed: Val and Test subjects overlap!"
    assert len(set_train | set_val | set_test) == 24, "Assertion failed: Subject count != 24!"
    assert len(split["train"]) == 17, f"Expected 17 train subjects, got {len(split['train'])}"
    assert len(split["val"]) == 3, f"Expected 3 val subjects, got {len(split['val'])}"
    assert len(split["test"]) == 4, f"Expected 4 test subjects, got {len(split['test'])}"
    print("  -> PASSED: Disjoint partition verified (17 train / 3 val / 4 test).")

    print("\n[Check 2] Windowing Array Dimensions & Counts:")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape} | y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape} | y_test:  {y_test.shape}")
    total_windows = len(y_train) + len(y_val) + len(y_test)
    print(f"  Total Extracted Windows: {total_windows:,}")

    assert X_train.shape[1:] == (window_size, 12), f"Unexpected X_train shape: {X_train.shape}"
    assert X_val.shape[1:] == (window_size, 12), f"Unexpected X_val shape: {X_val.shape}"
    assert X_test.shape[1:] == (window_size, 12), f"Unexpected X_test shape: {X_test.shape}"
    assert X_train.dtype == np.float32, f"Expected float32, got {X_train.dtype}"
    assert y_train.dtype == np.int64, f"Expected int64, got {y_train.dtype}"
    print("  -> PASSED: Array shapes and dtypes verified.")

    print("\n[Check 3] Boundary Guard (No window spans two trials):")
    for name, meta_list in [("Train", meta_train), ("Val", meta_val), ("Test", meta_test)]:
        for m in meta_list:
            span = m["end_sample_index"] - m["start_sample_index"] + 1
            # In a clean contiguous trial, end - start + 1 == window_size
            assert span == window_size, f"Corrupted window span in {name}: {m}"
            assert isinstance(m["subject_id"], int)
            assert isinstance(m["trial"], int)
            assert isinstance(m["source_file"], str)
    print("  -> PASSED: All windows strictly contained within individual trial bounds.")

    print("\n[Check 4] Leak-Free Scaler Verification:")
    scaler = joblib.load(DEFAULT_SCALER_PATH)
    # Scaler mean must match training feature mean before standardization
    print(f"  Fitted Scaler Feature Count: {scaler.n_features_in_}")
    print(f"  Scaler Mean (first 3 features): {scaler.mean_[:3]}")
    print(f"  Scaler Variance (first 3):     {scaler.var_[:3]}")

    # Re-verify training features transformed with this scaler have mean ≈ 0
    df = pd.read_csv(DEFAULT_CSV_PATH)
    col_map = detect_columns(list(df.columns))
    train_raw = df.loc[df[col_map["subject_id"]].isin(split["train"]), SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    train_transformed = scaler.transform(train_raw)
    train_feature_means = np.abs(train_transformed.mean(axis=0))
    train_feature_stds = train_transformed.std(axis=0)

    print(f"  Max Absolute Train Feature Mean after scaling: {train_feature_means.max():.2e}")
    print(f"  Mean Feature Std after scaling:                 {train_feature_stds.mean():.4f}")
    assert train_feature_means.max() < 1e-4, f"Scaler mean not centered at 0: {train_feature_means}"
    assert np.allclose(train_feature_stds, 1.0, atol=1e-3), f"Scaler variance not 1: {train_feature_stds}"
    print("  -> PASSED: Zero-leakage scaler verified (train mean ≈ 0, std ≈ 1).")

    print("\n[Check 5] PyTorch Dataset & DataLoader Interface:")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=64,
        seed=42,
        window_size=window_size,
        stride=stride,
    )

    batch_x, batch_y = next(iter(train_loader))
    print(f"  Batch x shape: {batch_x.shape} (Expected: (64, 12, {window_size}))")
    print(f"  Batch y shape: {batch_y.shape} (Expected: (64,))")
    assert batch_x.shape == (64, 12, window_size), f"Unexpected batch x shape: {batch_x.shape}"
    assert batch_y.shape == (64,), f"Unexpected batch y shape: {batch_y.shape}"
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64
    print("  -> PASSED: PyTorch Dataset layout matches model input expectations.")

    print("\n[Check 6] Class Weight Computation (Optional Tool):")
    class_weights = compute_class_weights(y_train)
    print(f"  Computed Balanced Class Weights: {class_weights.tolist()}")
    assert len(class_weights) == NUM_ACTIVITIES
    assert (class_weights > 0).all()
    print("  -> PASSED: Class weight calculator operational.")

    print("\n[Check 7] Data Summary File Verification:")
    assert DEFAULT_SUMMARY_PATH.exists(), f"Summary JSON missing: {DEFAULT_SUMMARY_PATH}"
    with open(DEFAULT_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary_content = json.load(f)
    print(f"  Summary saved at: {DEFAULT_SUMMARY_PATH}")
    print(f"  Total Windows in Summary: {summary_content['total_windows_per_split']['total']:,}")
    print("  Class counts per split (Train):", summary_content["class_counts_per_split"]["train"])
    print("  -> PASSED: Summary JSON verified.")

    print("\n" + "=" * 78)
    print("  ALL DATA_LOADER SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionSense Windowing & DataLoader Pipeline (Member 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for DataLoader (default 64)")
    parser.add_argument("--window-size", type=int, default=100, help="Window length in samples (default 100)")
    parser.add_argument("--stride", type=int, default=50, help="Window stride in samples (default 50)")
    parser.add_argument("--seed", type=int, default=42, help="Seeded RNG for subject split (default 42)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default 0)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding windows from raw CSV")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_CSV_PATH), help="Source CSV file")
    parser.add_argument("--test", action="store_true", help="Run comprehensive verification suite")

    args = parser.parse_args()

    # Always execute self-test if called directly with --test or by default
    run_self_test()


if __name__ == "__main__":
    main()
