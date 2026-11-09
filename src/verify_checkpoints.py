#!.venv/Scripts/python.exe
"""
src/verify_checkpoints.py

Verification script for Phase 3/4 deliverables (Member 2).
Loads both trained model checkpoints (BaselineCNN and AdaptiveSensorFusion),
runs inference on a batch of test data, and verifies:
    1. Checkpoints load successfully from models/ without shape mismatch.
    2. Input shape contract (batch_size, 12, 100) is accepted.
    3. Output logits shape is strictly (batch_size, 6) with all finite values.
    4. Model predictions are 1D integer tensors with values in 0..5.
    5. AdaptiveSensorFusion returns dynamic branch weights summing to 1.0.
    6. Boundary masking works: missing_acc -> w_acc=0.0, w_gyro=1.0.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data_loader import get_dataloaders
from src.model import BaselineCNN, load_model
from src.sensor_degradation import forward_with_condition
from src.sensor_fusion import AdaptiveSensorFusion, load_adaptive_model
from src.train import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_checkpoints")

MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    print("=" * 78)
    print("  VERIFYING SAVED MODEL CHECKPOINTS & INFERENCE CONTRACT")
    print("=" * 78)

    device = torch.device("cpu")  # Portable CPU verification

    # 1. Verify files exist
    baseline_pth = MODELS_DIR / "baseline_best.pth"
    adaptive_pth = MODELS_DIR / "adaptive_best.pth"
    baseline_cfg = MODELS_DIR / "baseline_config.json"
    adaptive_cfg = MODELS_DIR / "adaptive_config.json"
    scaler_pkl = MODELS_DIR / "train_only_scaler.pkl"

    for p in [baseline_pth, adaptive_pth, baseline_cfg, adaptive_cfg, scaler_pkl]:
        assert p.exists(), f"Missing required file: {p}"
        print(f"  [OK] Found: {p.name} ({p.stat().st_size:,} bytes)")

    # 2. Read configs for constructor arguments
    with open(baseline_cfg, "r", encoding="utf-8") as f:
        cfg_base = json.load(f)
    with open(adaptive_cfg, "r", encoding="utf-8") as f:
        cfg_adapt = json.load(f)

    dropout_base = float(cfg_base["hyperparameters"]["dropout"])
    dropout_adapt = float(cfg_adapt["hyperparameters"]["dropout"])

    print(f"\nConstructor Hyperparameters:")
    print(f"  - Baseline: dropout={dropout_base}, lr={cfg_base['hyperparameters']['lr']}, opt={cfg_base['hyperparameters']['optimizer']}")
    print(f"  - Adaptive: dropout={dropout_adapt}, lr={cfg_adapt['hyperparameters']['lr']}, opt={cfg_adapt['hyperparameters']['optimizer']}")

    # 3. Instantiate and load models
    print("\nLoading Model Checkpoints:")
    model_base = build_model("baseline", dropout=dropout_base).to(device)
    load_model(model_base, str(baseline_pth), map_location=str(device))
    model_base.eval()
    print("  [OK] Successfully loaded BaselineCNN from models/baseline_best.pth")

    model_adapt = build_model("adaptive", dropout=dropout_adapt).to(device)
    load_adaptive_model(model_adapt, str(adaptive_pth), map_location=str(device))
    model_adapt.eval()
    print("  [OK] Successfully loaded AdaptiveSensorFusion from models/adaptive_best.pth")

    # 4. Fetch one test batch from DataLoader
    print("\nFetching Test Batch via get_dataloaders():")
    _, _, test_loader = get_dataloaders(batch_size=16, seed=42, num_workers=0)
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    print(f"  - Test batch x shape: {tuple(batch_x.shape)} (Expected: (16, 12, 100))")
    print(f"  - Test batch y shape: {tuple(batch_y.shape)} (Expected: (16,))")
    assert batch_x.shape == (16, 12, 100), f"Unexpected input shape: {batch_x.shape}"
    assert batch_x.dtype == torch.float32, f"Expected float32, got {batch_x.dtype}"

    # 5. Verify Baseline forward pass
    print("\nVerifying BaselineCNN Inference:")
    with torch.no_grad():
        logits_base = model_base(batch_x)
        preds_base = model_base.predict(batch_x)

    print(f"  - Logits shape: {tuple(logits_base.shape)} (Expected: (16, 6))")
    print(f"  - Predictions shape: {tuple(preds_base.shape)} (Expected: (16,))")
    assert logits_base.shape == (16, 6), f"Wrong logits shape: {logits_base.shape}"
    assert torch.isfinite(logits_base).all(), "Logits contain non-finite values (NaN/Inf)"
    assert preds_base.shape == (16,), f"Wrong preds shape: {preds_base.shape}"
    assert (preds_base >= 0).all() and (preds_base < 6).all(), "Predictions out of class range 0..5"
    print("  [OK] BaselineCNN forward pass, predict(), and shape assertions passed.")

    # 6. Verify AdaptiveSensorFusion forward pass & dynamic weights
    print("\nVerifying AdaptiveSensorFusion Inference & Dynamic Weights:")
    with torch.no_grad():
        mask_ones = torch.ones(16, dtype=torch.float32, device=device)
        logits_adapt, w_info = model_adapt(batch_x, mask_acc=mask_ones, mask_gyro=mask_ones, return_weights=True)
        preds_adapt = model_adapt.predict(batch_x, mask_acc=mask_ones, mask_gyro=mask_ones)

    print(f"  - Logits shape: {tuple(logits_adapt.shape)} (Expected: (16, 6))")
    print(f"  - Predictions shape: {tuple(preds_adapt.shape)} (Expected: (16,))")
    assert logits_adapt.shape == (16, 6), f"Wrong logits shape: {logits_adapt.shape}"
    assert torch.isfinite(logits_adapt).all(), "Logits contain non-finite values (NaN/Inf)"
    assert preds_adapt.shape == (16,), f"Wrong preds shape: {preds_adapt.shape}"
    assert (preds_adapt >= 0).all() and (preds_adapt < 6).all(), "Predictions out of class range 0..5"

    w_acc = w_info["w_acc"].flatten()
    w_gyro = w_info["w_gyro"].flatten()
    weight_sum = w_acc + w_gyro
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), "Weights do not sum to 1.0"
    print(f"  - Mean weights (clean): w_acc={w_acc.mean().item():.4f}, w_gyro={w_gyro.mean().item():.4f} (Sum={weight_sum.mean().item():.4f})")
    print("  [OK] AdaptiveSensorFusion standard inference and weight sum verified.")

    # 7. Verify Boundary Condition: missing_acc
    print("\nVerifying Degradation Boundary Condition (missing_acc):")
    with torch.no_grad():
        preds_missing, w_missing = forward_with_condition(
            model=model_adapt,
            x=batch_x,
            condition="missing_acc",
            seed=42,
            return_weights=True,
            mode="predict",
        )
    w_acc_m = w_missing["w_acc"].flatten()
    w_gyro_m = w_missing["w_gyro"].flatten()
    assert torch.allclose(w_acc_m, torch.zeros_like(w_acc_m)), "w_acc must be exactly 0.0 under missing_acc!"
    assert torch.allclose(w_gyro_m, torch.ones_like(w_gyro_m)), "w_gyro must be exactly 1.0 under missing_acc!"
    print(f"  - Under missing_acc: w_acc={w_acc_m.mean().item():.1f}, w_gyro={w_gyro_m.mean().item():.1f}")
    print("  [OK] Boundary condition verified: 100% reliance dynamically shifted to Gyroscope branch.")

    # 8. Check degradation-aware checkpoints if present
    deg_base_pth = MODELS_DIR / "baseline_degradation_aware.pth"
    deg_adapt_pth = MODELS_DIR / "adaptive_degradation_aware.pth"
    if deg_base_pth.exists() and deg_adapt_pth.exists():
        print("\nVerifying Degradation-Aware Checkpoints:")
        m_b_deg = build_model("baseline", dropout=dropout_base).to(device)
        load_model(m_b_deg, str(deg_base_pth), map_location=str(device))
        assert m_b_deg(batch_x).shape == (16, 6)
        print("  [OK] baseline_degradation_aware.pth loaded and verified.")

        m_a_deg = build_model("adaptive", dropout=dropout_adapt).to(device)
        load_adaptive_model(m_a_deg, str(deg_adapt_pth), map_location=str(device))
        assert m_a_deg(batch_x, mask_acc=mask_ones, mask_gyro=mask_ones).shape == (16, 6)
        print("  [OK] adaptive_degradation_aware.pth loaded and verified.")

    print("\n" + "=" * 78)
    print("  ALL CHECKPOINT VERIFICATIONS PASSED SUCCESSFULLY! (Output shape: (B, 6))")
    print("=" * 78)


if __name__ == "__main__":
    main()

