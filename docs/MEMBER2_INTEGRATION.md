# Phase 3/4 Deliverables & Integration Guide (Member 2)

**Author:** Member 2  
**Project:** Adaptive Sensor Fusion for Human Activity Recognition (MotionSense)  
**Deliverables Scope:** Data windowing, subject-wise splitting, training pipeline, hyperparameter tuning, sensor degradation benchmarking, and LaTeX reporting.

---

## 1. Folder Structure of Deliverables

```
adaptive-sensor-fusion/
├── docs/
│   └── MEMBER2_INTEGRATION.md                 # This integration reference document
├── models/
│   ├── baseline_best.pth                      # Best BaselineCNN checkpoint (clean training)
│   ├── baseline_config.json                   # Hyperparameters and metadata for baseline_best
│   ├── adaptive_best.pth                      # Best AdaptiveSensorFusion checkpoint (clean training)
│   ├── adaptive_config.json                   # Hyperparameters and metadata for adaptive_best
│   ├── baseline_degradation_aware.pth         # BaselineCNN checkpoint (robust augmentation)
│   ├── baseline_degradation_aware_config.json # Hyperparameters for degradation-aware baseline
│   ├── adaptive_degradation_aware.pth         # AdaptiveSensorFusion checkpoint (robust augmentation)
│   ├── adaptive_degradation_aware_config.json # Hyperparameters for degradation-aware adaptive
│   └── train_only_scaler.pkl                  # Leak-free StandardScaler fitted on training subjects only
├── motionsense_output/
│   ├── activity_label_mapping.json            # Target activity label integer mapping (0-5)
│   ├── motionsense_preprocessed.csv           # Phase 2 preprocessed time-series dataset (628 MB)
│   └── windows_w100_s50_seed42.npz            # Cached temporal window arrays (27,720 windows)
├── results/
│   ├── subject_split.json                     # Disjoint 70/15/15 subject partition definitions
│   ├── data_summary.json                      # Window counts and per-class distributions per split
│   ├── best_hyperparameters.json              # Selected optimal hyperparameters per architecture
│   ├── hyperparameter_results.csv             # Full hyperparameter sweep logs (40 evaluated runs)
│   ├── hyperparameter_summary.png             # Tuning validation accuracy heatmap by LR x Batch Size
│   ├── degradation_results.csv                # Accuracy, Precision, Recall, F1 across 6 conditions
│   ├── noise_sweep.csv                        # Performance trajectory over sigma in [0.0, 2.0]
│   ├── adaptive_fusion_weights.csv            # Dynamic fusion branch weights (w_acc, w_gyro)
│   ├── degradation_accuracy_comparison.png    # Publication grouped bar chart: Accuracy across conditions
│   ├── degradation_f1_comparison.png          # Publication grouped bar chart: Macro F1 across conditions
│   ├── noise_sweep.png                        # 3-panel noise sensitivity line trajectories
│   ├── confusion_matrix_normal_baseline.png   # Baseline test confusion matrix (clean data)
│   ├── confusion_matrix_normal_adaptive.png   # Adaptive test confusion matrix (clean data)
│   ├── adaptive_fusion_weights.png            # Dynamic weight reallocation visualization
│   ├── loss_curve.png                         # Training & validation CrossEntropy loss trajectory
│   ├── accuracy_curve.png                     # Training & validation classification accuracy trajectory
│   └── training_history.csv                   # Per-epoch training & validation history log
├── Reports/
│   ├── training_and_tuning.tex                # LaTeX section fragment (\input-ready)
│   └── tables/
│       ├── split_table.tex                    # Auto-generated participant & window split table
│       ├── hparam_table.tex                   # Auto-generated hyperparameter selection table
│       ├── degradation_table.tex              # Auto-generated 6-condition test performance table
│       └── weights_table.tex                  # Auto-generated dynamic fusion weights table
└── src/
    ├── data_loader.py                         # Ingestion, scaling, splitting, and window generation
    ├── sensor_degradation.py                  # Non-destructive degradation simulation & wrapper interface
    ├── train.py                               # Model factory, training engine, early stopping, curves
    ├── hyperparameter_tuning.py               # Stratified random & staged hyperparameter search
    ├── run_experiments.py                     # Orchestrator: training, evaluation, sweeps, plotting
    ├── verify_checkpoints.py                  # Integration verification script
    └── generate_latex_tables.py               # Table generation directly from CSV/JSON artifacts
```

---

## 2. How to Load Each Trained Model

Member 1's models are instantiated with their exact constructor arguments (retrieved from `models/*_config.json`) and populated using the official checkpoint helpers.

```python
import json
import torch
from src.model import BaselineCNN, load_model
from src.sensor_fusion import AdaptiveSensorFusion, load_adaptive_model
from src.train import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# 1. Load Baseline 1D CNN Checkpoint
# ----------------------------------------------------------------------------
with open("models/baseline_config.json", "r") as f:
    base_cfg = json.load(f)

# Read tuned dropout rate (0.3)
dropout_base = float(base_cfg["hyperparameters"]["dropout"])

# Instantiate and load weights
model_base = build_model("baseline", dropout=dropout_base).to(device)
load_model(model_base, "models/baseline_best.pth", map_location=str(device))
model_base.eval()

# ----------------------------------------------------------------------------
# 2. Load Adaptive Sensor Fusion Checkpoint
# ----------------------------------------------------------------------------
with open("models/adaptive_config.json", "r") as f:
    adapt_cfg = json.load(f)

# Read tuned dropout rate (0.5)
dropout_adapt = float(adapt_cfg["hyperparameters"]["dropout"])

# Instantiate and load weights
model_adapt = build_model("adaptive", dropout=dropout_adapt).to(device)
load_adaptive_model(model_adapt, "models/adaptive_best.pth", map_location=str(device))
model_adapt.eval()

# ----------------------------------------------------------------------------
# 3. Load Degradation-Aware Checkpoints (Optional)
# ----------------------------------------------------------------------------
model_base_robust = build_model("baseline", dropout=dropout_base).to(device)
load_model(model_base_robust, "models/baseline_degradation_aware.pth", map_location=str(device))
model_base_robust.eval()

model_adapt_robust = build_model("adaptive", dropout=dropout_adapt).to(device)
load_adaptive_model(model_adapt_robust, "models/adaptive_degradation_aware.pth", map_location=str(device))
model_adapt_robust.eval()
```

---

## 3. Input Contract & Sensor Specifications

### Tensor Contract
- **Tensor Shape**: `(batch_size, 12, 100)` $\to$ `(B, C, L)`
  - $B$: Mini-batch size (e.g. 16, 32, 64)
  - $C = 12$: Continuous sensor feature channels
  - $L = 100$: Sequence length ($100 \text{ readings} \times 20\text{ ms} = 2.0 \text{ seconds}$ at 50 Hz)
- **Data Type**: `torch.float32` (standardized features with zero mean and unit variance)

### Feature Channel Ordering (Exact Order)
```
Index 0: attitude.roll        (Orientation / Euler angle)
Index 1: attitude.pitch       (Orientation / Euler angle)
Index 2: attitude.yaw         (Orientation / Euler angle)
Index 3: gravity.x            (Directional Gravity Vector)
Index 4: gravity.y            (Directional Gravity Vector)
Index 5: gravity.z            (Directional Gravity Vector)
Index 6: rotationRate.x       (Angular Velocity)
Index 7: rotationRate.y       (Angular Velocity)
Index 8: rotationRate.z       (Angular Velocity)
Index 9: userAcceleration.x   (Linear Dynamic Acceleration)
Index 10: userAcceleration.y  (Linear Dynamic Acceleration)
Index 11: userAcceleration.z  (Linear Dynamic Acceleration)
```

### Branch Channel Partitioning Indices
Exported in `src/__init__.py`:
- `ACC_INDICES = [3, 4, 5, 9, 10, 11]` (Gravity + User Acceleration: 6 features $\to$ Branch 1)
- `GYRO_INDICES = [0, 1, 2, 6, 7, 8]` (Attitude Euler angles + Rotation Rate: 6 features $\to$ Branch 2)

### Train-Only StandardScaler (`models/train_only_scaler.pkl`)
- **Leak-Free Guarantee**: The Phase 2 scaler was fitted across all 24 participants, leaking test-set distribution statistics. Member 2 fitted a new `sklearn.preprocessing.StandardScaler` **strictly** on the 17 training participants.
- To standardize unscaled raw sensor arrays $X_{\text{raw}} \in \mathbb{R}^{N \times 12}$:
```python
import joblib

scaler = joblib.load("models/train_only_scaler.pkl")
# Standardize raw 12-channel data
X_scaled = scaler.transform(X_raw)
```

### Target Class Mapping (`motionsense_output/activity_label_mapping.json`)
Alphabetical integer encoding (0 to 5):
```json
{
  "downstairs": 0,
  "jogging": 1,
  "sitting": 2,
  "standing": 3,
  "upstairs": 4,
  "walking": 5
}
```

---

## 4. How Member 3 Loads the Identical Split

Member 3 must use `src.data_loader.load_split()` or `get_dataloaders()` to ensure zero cross-contamination and bit-identical subject partitioning.

### Participant Partitioning (`results/subject_split.json`)
- **Train (17 participants, ~70.83%)**:
  `[1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 16, 17, 18, 19, 22, 23]`
- **Validation (3 participants, ~12.50%)**:
  `[8, 21, 24]`
- **Test (4 participants, ~16.67%)**:
  `[7, 11, 15, 20]`

### Usage Snippet for Member 3
```python
from src.data_loader import get_dataloaders, load_split

# 1. Load exact participant ID lists
split_info = load_split("results/subject_split.json")
print("Train participants:", split_info["train_subjects"])
print("Val participants:  ", split_info["val_subjects"])
print("Test participants: ", split_info["test_subjects"])

# 2. Instantiate PyTorch DataLoaders (returns x shaped (12, 100))
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=64,
    seed=42,
    num_workers=0,
)

# Verify first batch
batch_x, batch_y = next(iter(test_loader))
print("Test batch x shape:", batch_x.shape)  # torch.Size([64, 12, 100])
print("Test batch y shape:", batch_y.shape)  # torch.Size([64])
```

---

## 5. Applying Sensor Degradation & Availability Masks

The degradation simulation engine in `src/sensor_degradation.py` is strictly **non-destructive** (always clones tensors) and seeded for determinism.

### Available Physical Conditions
- `normal`: Clean baseline signal.
- `noisy_acc`: Gaussian noise on accelerometer channels (default $\sigma = 0.5$).
- `noisy_gyro`: Gaussian noise on gyroscope channels (default $\sigma = 0.5$).
- `missing_acc`: Accelerometer channels zeroed out; `mask_acc = 0.0`.
- `missing_gyro`: Gyroscope channels zeroed out; `mask_gyro = 0.0`.
- `noisy_both`: Elevated Gaussian noise on all 12 channels ($\sigma = 1.0$).

### Usage Snippet: Unified Forward Wrapper
```python
from src.sensor_degradation import forward_with_condition, apply_condition

# Method 1: Using the universal evaluation wrapper (handles both architectures)
with torch.no_grad():
    # Returns predicted class indices (0-5) under complete accelerometer failure
    preds = forward_with_condition(
        model=model_adapt,
        x=batch_x,
        condition="missing_acc",
        seed=42,
        mode="predict",
    )

    # For adaptive model, inspect dynamic weights reallocation
    preds, weights = forward_with_condition(
        model=model_adapt,
        x=batch_x,
        condition="missing_acc",
        seed=42,
        return_weights=True,
        mode="predict",
    )
    print("w_acc mean: ", weights["w_acc"].mean().item())   # Strictly 0.0
    print("w_gyro mean:", weights["w_gyro"].mean().item())  # Strictly 1.0

# Method 2: Manual condition application & direct model invocation
x_degraded, masks = apply_condition(batch_x, condition="noisy_acc", sigma=0.5, seed=42)

# Pass availability masks to AdaptiveSensorFusion
logits = model_adapt(
    x=x_degraded,
    mask_acc=masks["mask_acc"],
    mask_gyro=masks["mask_gyro"],
    reliability_acc=masks["reliability_acc"],
    reliability_gyro=masks["reliability_gyro"],
)
```

---

## 6. Commands to Reproduce Everything

Run all commands from the repository root:

```powershell
# 1. Verify Checkpoints and Inference Contract
py src/verify_checkpoints.py

# 2. Run Hyperparameter Tuning (Stratified Random Sweep, 20 runs per model)
py src/hyperparameter_tuning.py --mode random --max-runs 20 --models both

# 3. Execute Complete Experiments Pipeline (Training, Evaluation, Sigma Sweep, Plots, Summary)
py src/run_experiments.py --all

# 4. Generate LaTeX Tables from Saved CSVs
py src/generate_latex_tables.py

# 5. Quick Smoke Test (Verifies complete pipeline in ~15 seconds)
py src/run_experiments.py --smoke
```

