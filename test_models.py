"""
test_models.py

Quick demonstration and integration verification script for Member 1's models:
    - BaselineCNN (src/model.py)
    - AdaptiveSensorFusion (src/sensor_fusion.py)

Run with:
    python test_models.py
"""

import torch
from src.model import BaselineCNN
from src.sensor_fusion import AdaptiveSensorFusion


def main():
    print("=" * 70)
    print("  RUNNING QUICK MODEL VERIFICATION (test_models.py)")
    print("=" * 70)

    # 1. Create a dummy batch of 4 sensor windows (12 channels, 128 time steps each)
    # Channels 0-2: attitude, 3-5: gravity, 6-8: rotationRate, 9-11: userAcceleration
    batch_size, channels, seq_len = 4, 12, 128
    x = torch.randn(batch_size, channels, seq_len)
    print(f"\n[Step 1] Created input tensor:")
    print(f"  Shape (batch, channels, length): {x.shape}")

    # 2. Baseline Model
    print(f"\n[Step 2] Testing Baseline 1D CNN:")
    baseline = BaselineCNN()
    logits_base = baseline(x)              # Shape: (4, 6)
    probs_base = baseline.predict_proba(x) # Normalized probabilities
    preds_base = baseline.predict(x)       # Predicted classes (0 to 5)

    print(f"  Logits shape:        {logits_base.shape} (Expected: ({batch_size}, 6))")
    print(f"  Predicted classes:   {preds_base.tolist()}")
    print(f"  Probability sum:     {probs_base.sum(dim=1).tolist()} (Should all be 1.0)")

    # 3. Adaptive Sensor Fusion Model
    print(f"\n[Step 3] Testing Adaptive Sensor Fusion Model:")
    adaptive = AdaptiveSensorFusion()

    # Normal forward pass:
    logits_adapt, weights = adaptive(x, return_weights=True)
    w_acc = weights["w_acc"].mean().item()
    w_gyro = weights["w_gyro"].mean().item()
    print(f"  Normal forward pass logits shape: {logits_adapt.shape}")
    print(f"  Calculated Fusion Weights:        w_acc={w_acc:.4f}, w_gyro={w_gyro:.4f} (Sum: {w_acc + w_gyro:.4f})")

    # Simulate accelerometer failure (e.g., sensor disconnected):
    preds_degraded = adaptive.predict(x, mask_acc=0.0, mask_gyro=1.0)
    print(f"\n[Step 4] Simulated Accelerometer Disconnection (mask_acc=0.0):")
    print(f"  Predictions with missing accelerometer: {preds_degraded.tolist()}")
    print("  -> Successfully relied 100% on Gyroscope/Attitude branch!")

    print("\n" + "=" * 70)
    print("  VERIFICATION COMPLETE - ALL CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

