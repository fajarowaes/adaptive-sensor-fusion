#!.venv/Scripts/python.exe
"""
src/sensor_degradation.py

Phase 3/4 Sensor Degradation & Robustness Simulation Engine for Human Activity Recognition.
Owned by Member 2.

Key Responsibilities:
    1. Condition Registry:
       Defines reproducible sensor degradation conditions:
       - 'normal': Pristine standardized sensor stream
       - 'noisy_acc': Gaussian noise added exclusively to accelerometer/gravity channels
       - 'noisy_gyro': Gaussian noise added exclusively to gyroscope/attitude channels
       - 'missing_acc': Accelerometer stream disconnected (zeroed channels + mask_acc=0.0)
       - 'missing_gyro': Gyroscope stream disconnected (zeroed channels + mask_gyro=0.0)
       - 'noisy_both': Elevated Gaussian noise on all 12 sensor channels
    2. Non-Destructive Invariant:
       Never mutates input tensors in-place; always clones input tensors before applying corruptions.
    3. Strict Reproducibility:
       Guaranteed deterministic noise generation via seeded torch.Generator instances
       keyed by (condition, seed).
    4. Model Wrapper Interface:
       Unified forward_with_condition() seamlessly dispatching inputs, masks, and reliability
       indicators to either BaselineCNN or AdaptiveSensorFusion.
    5. Degradation-Aware Training Augmentation:
       degrade_batch_random() for optional robust training mode with per-sample stochastic masks.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path to enable execution as standalone script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.model import BaselineCNN
from src.sensor_fusion import ACC_INDICES, GYRO_INDICES, AdaptiveSensorFusion

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sensor_degradation")

# ============================================================================
# Degradation Registry & Parameters
# ============================================================================
SIGMA_SWEEP_VALUES: List[float] = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


@dataclass(frozen=True)
class DegradationCondition:
    """Specification of a physical sensor degradation condition."""

    name: str
    noise_channels: Optional[Tuple[int, ...]] = None
    missing_channels: Optional[Tuple[int, ...]] = None
    default_sigma: float = 0.0
    mask_acc_default: float = 1.0
    mask_gyro_default: float = 1.0
    description: str = ""


DEGRADATION_REGISTRY: Dict[str, DegradationCondition] = {
    "normal": DegradationCondition(
        name="normal",
        noise_channels=None,
        missing_channels=None,
        default_sigma=0.0,
        mask_acc_default=1.0,
        mask_gyro_default=1.0,
        description="Pristine condition: all 12 channels available with zero synthetic degradation.",
    ),
    "noisy_acc": DegradationCondition(
        name="noisy_acc",
        noise_channels=tuple(ACC_INDICES),
        missing_channels=None,
        default_sigma=0.5,
        mask_acc_default=1.0,
        mask_gyro_default=1.0,
        description="Accelerometer corrupted: zero-mean Gaussian noise on ACC channels (indices 3,4,5,9,10,11).",
    ),
    "noisy_gyro": DegradationCondition(
        name="noisy_gyro",
        noise_channels=tuple(GYRO_INDICES),
        missing_channels=None,
        default_sigma=0.5,
        mask_acc_default=1.0,
        mask_gyro_default=1.0,
        description="Gyroscope corrupted: zero-mean Gaussian noise on GYRO channels (indices 0,1,2,6,7,8).",
    ),
    "missing_acc": DegradationCondition(
        name="missing_acc",
        noise_channels=None,
        missing_channels=tuple(ACC_INDICES),
        default_sigma=0.0,
        mask_acc_default=0.0,
        mask_gyro_default=1.0,
        description="Complete accelerometer failure: ACC channels zeroed out; mask_acc=0.0.",
    ),
    "missing_gyro": DegradationCondition(
        name="missing_gyro",
        noise_channels=None,
        missing_channels=tuple(GYRO_INDICES),
        default_sigma=0.0,
        mask_acc_default=1.0,
        mask_gyro_default=0.0,
        description="Complete gyroscope failure: GYRO channels zeroed out; mask_gyro=0.0.",
    ),
    "noisy_both": DegradationCondition(
        name="noisy_both",
        noise_channels=tuple(sorted(ACC_INDICES + GYRO_INDICES)),
        missing_channels=None,
        default_sigma=1.0,
        mask_acc_default=1.0,
        mask_gyro_default=1.0,
        description="Severe environmental noise: elevated Gaussian noise across all 12 sensor channels.",
    ),
}

DEGRADATION_CONDITIONS: List[str] = list(DEGRADATION_REGISTRY.keys())


# ============================================================================
# Deterministic Generator Helper
# ============================================================================
def get_seeded_generator(
    condition: str,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = "cpu",
) -> torch.Generator:
    """
    Constructs a deterministic torch.Generator seeded per (condition, seed).

    Ensures every model architecture and evaluation pass encounters
    bit-identical synthetic noise perturbations.
    """
    gen_device = torch.device(device) if device is not None else torch.device("cpu")
    # torch.Generator device must be 'cpu' for cpu tensors or 'cuda' for cuda tensors
    gen = torch.Generator(device="cpu" if gen_device.type == "cpu" else gen_device)

    if seed is not None:
        # Create deterministic condition hash offset
        condition_offset = sum((i + 1) * ord(c) for i, c in enumerate(condition))
        derived_seed = (int(seed) * 10007 + condition_offset) % (2**31 - 1)
        gen.manual_seed(derived_seed)

    return gen


# ============================================================================
# Core Degradation Engine (Non-Destructive)
# ============================================================================
def apply_condition(
    x: torch.Tensor,
    condition: str,
    sigma: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = 42,
    compute_reliability: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    r"""
    Applies reproducible, non-destructive sensor degradation to a standardized batch.

    Mathematical Formulation:
        - Clean standardized sensor features have zero mean and unit variance ($E[X]=0, \text{Var}(X)=1$).
        - Missing channels are set to 0.0 (the standardized training set mean) to simulate inactive
          sensor bus output without inducing extreme numerical spikes.
        - Additive Gaussian noise: $X_{\text{noisy}} = X + \mathcal{N}(0, \sigma^2 \cdot \mathbf{I})$.
        - Reliability Indicator (when compute_reliability=True):
          Modeled from signal-to-noise ratio: $r = \frac{1}{1 + \sigma^2} \in (0, 1]$.
          When compute_reliability=False (default setting), reliability is None, prompting
          AdaptiveSensorFusion to invoke its learned internal self-confidence estimator.

    Args:
        x: Input sensor tensor of shape (batch_size, 12, sequence_length).
        condition: Key from DEGRADATION_REGISTRY.
        sigma: Standard deviation of Gaussian noise. If None, uses condition's default.
        generator: Optional seeded torch.Generator.
        seed: Random seed used if generator is None (default 42).
        compute_reliability: If True, provides analytical SNR-based reliability in masks.
                             Default False (let model self-estimate).

    Returns:
        x_degraded: New tensor with degradation applied (input tensor is never mutated).
        masks: Dictionary containing:
            - 'mask_acc': Tensor of shape (batch_size,), dtype=torch.float32.
            - 'mask_gyro': Tensor of shape (batch_size,), dtype=torch.float32.
            - 'reliability_acc': Optional Tensor of shape (batch_size,) or None.
            - 'reliability_gyro': Optional Tensor of shape (batch_size,) or None.
    """
    if condition not in DEGRADATION_REGISTRY:
        raise ValueError(
            f"Unknown degradation condition '{condition}'. "
            f"Available options: {list(DEGRADATION_REGISTRY.keys())}"
        )

    spec = DEGRADATION_REGISTRY[condition]
    batch_size = x.size(0)
    device = x.device
    dtype = x.dtype

    # ALWAYS clone: strictly non-destructive guarantee
    x_degraded = x.clone()

    noise_std = spec.default_sigma if sigma is None else float(sigma)

    # 1. Apply Additive Gaussian Noise
    if spec.noise_channels is not None and noise_std > 0.0:
        if generator is None:
            gen = get_seeded_generator(condition=condition, seed=seed, device=device)
        else:
            gen = generator

        channels = list(spec.noise_channels)
        target_shape = (batch_size, len(channels), x.size(2))

        # Generate zero-mean Gaussian noise on CPU/device matching generator
        if gen.device.type == "cpu" and device.type != "cpu":
            noise = torch.randn(target_shape, generator=gen, dtype=dtype).to(device)
        else:
            noise = torch.randn(target_shape, generator=gen, device=device, dtype=dtype)

        x_degraded[:, channels, :] = x_degraded[:, channels, :] + (noise_std * noise)

    # 2. Apply Sensor Disconnection / Missing Modality
    if spec.missing_channels is not None:
        channels = list(spec.missing_channels)
        # Missing sensor reads zero (standardized mean)
        x_degraded[:, channels, :] = 0.0

    # 3. Construct Availability Masks matching Member 1's expected shape: (B,) float32
    mask_acc = torch.full((batch_size,), spec.mask_acc_default, dtype=torch.float32, device=device)
    mask_gyro = torch.full((batch_size,), spec.mask_gyro_default, dtype=torch.float32, device=device)

    # 4. Optional Analytical Reliability Scores (Default None -> Learned Confidence)
    reliability_acc: Optional[torch.Tensor] = None
    reliability_gyro: Optional[torch.Tensor] = None

    if compute_reliability:
        # Analytical reliability derived from SNR: r = 1 / (1 + sigma^2)
        r_acc_val = 0.0 if spec.mask_acc_default == 0.0 else 1.0 / (1.0 + (noise_std**2 if spec.name in ("noisy_acc", "noisy_both") else 0.0))
        r_gyro_val = 0.0 if spec.mask_gyro_default == 0.0 else 1.0 / (1.0 + (noise_std**2 if spec.name in ("noisy_gyro", "noisy_both") else 0.0))

        reliability_acc = torch.full((batch_size,), r_acc_val, dtype=torch.float32, device=device)
        reliability_gyro = torch.full((batch_size,), r_gyro_val, dtype=torch.float32, device=device)

    masks: Dict[str, Optional[torch.Tensor]] = {
        "mask_acc": mask_acc,
        "mask_gyro": mask_gyro,
        "reliability_acc": reliability_acc,
        "reliability_gyro": reliability_gyro,
    }

    return x_degraded, masks


# ============================================================================
# Universal Model Forward Wrapper
# ============================================================================
def is_adaptive_fusion_model(model: nn.Module) -> bool:
    """Detects whether a model is an AdaptiveSensorFusion instance."""
    if isinstance(model, AdaptiveSensorFusion):
        return True
    if hasattr(model, "fusion_module") and hasattr(model, "acc_branch"):
        return True
    # Check forward signature for adaptive arguments
    sig = inspect.signature(model.forward)
    return "mask_acc" in sig.parameters and "mask_gyro" in sig.parameters


def forward_with_condition(
    model: nn.Module,
    x: torch.Tensor,
    condition: str = "normal",
    sigma: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = 42,
    compute_reliability: bool = False,
    return_weights: bool = False,
    mode: str = "forward",
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Executes a model evaluation forward pass under specified sensor degradation.

    Auto-Detects Architecture:
        - BaselineCNN: Receives x_degraded. Ignores availability masks as it lacks
          branch-specific isolation.
        - AdaptiveSensorFusion: Receives x_degraded along with mask_acc, mask_gyro,
          and optional reliability scores, triggering dynamic weight redistribution.

    Args:
        model: PyTorch model instance (BaselineCNN or AdaptiveSensorFusion).
        x: Input tensor of shape (batch_size, 12, sequence_length).
        condition: Degradation condition key.
        sigma: Noise standard deviation override.
        generator: Optional seeded torch.Generator.
        seed: Seed for reproducibility if generator is None.
        compute_reliability: If True, provides external SNR reliability to model.
        return_weights: If True (and model supports it), returns (logits/preds, weights_dict).
        mode: Execution mode: 'forward' (logits), 'predict' (classes), or 'predict_proba' (probabilities).

    Returns:
        Predictions/logits tensor, or tuple of (tensor, weights_info) if return_weights=True.
    """
    x_degraded, masks = apply_condition(
        x=x,
        condition=condition,
        sigma=sigma,
        generator=generator,
        seed=seed,
        compute_reliability=compute_reliability,
    )

    is_adaptive = is_adaptive_fusion_model(model)

    if mode == "predict":
        if is_adaptive:
            preds = model.predict(
                x=x_degraded,
                mask_acc=masks["mask_acc"],
                mask_gyro=masks["mask_gyro"],
                reliability_acc=masks["reliability_acc"],
                reliability_gyro=masks["reliability_gyro"],
            )
        else:
            preds = model.predict(x_degraded)

        if return_weights and is_adaptive:
            _, weights = model(
                x=x_degraded,
                mask_acc=masks["mask_acc"],
                mask_gyro=masks["mask_gyro"],
                reliability_acc=masks["reliability_acc"],
                reliability_gyro=masks["reliability_gyro"],
                return_weights=True,
            )
            return preds, weights
        return preds

    elif mode == "predict_proba":
        if is_adaptive:
            probs = model.predict_proba(
                x=x_degraded,
                mask_acc=masks["mask_acc"],
                mask_gyro=masks["mask_gyro"],
                reliability_acc=masks["reliability_acc"],
                reliability_gyro=masks["reliability_gyro"],
            )
        else:
            probs = model.predict_proba(x_degraded)
        return probs

    elif mode == "forward":
        if is_adaptive:
            return model(
                x=x_degraded,
                mask_acc=masks["mask_acc"],
                mask_gyro=masks["mask_gyro"],
                reliability_acc=masks["reliability_acc"],
                reliability_gyro=masks["reliability_gyro"],
                return_weights=return_weights,
            )
        else:
            out = model(x_degraded)
            if return_weights:
                return out, {}
            return out

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from 'forward', 'predict', or 'predict_proba'.")


# ============================================================================
# Degradation-Aware Training Augmentation (Optional)
# ============================================================================
def degrade_batch_random(
    x: torch.Tensor,
    p_noise: float = 0.0,
    p_missing: float = 0.0,
    sigma: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Stochastic batch degradation for OPTIONAL 'degradation-aware' training mode.

    Features:
        - Completely decoupled from standard clean training path.
        - Applies vectorized, sample-wise degradation across mini-batch.
        - Each sample independently samples:
          * Normal pristine: 1 - (p_noise + p_missing)
          * Missing modality: p_missing (randomly missing_acc or missing_gyro)
          * Noisy modality: p_noise (randomly noisy_acc, noisy_gyro, or noisy_both)
        - Vectorized availability masks correctly reflect each sample's state.

    Args:
        x: Mini-batch tensor of shape (batch_size, 12, sequence_length).
        p_noise: Probability of applying Gaussian noise (default 0.0).
        p_missing: Probability of applying sensor dropout (default 0.0).
        sigma: Standard deviation of noise when noise is selected (default 0.5).
        generator: Optional torch.Generator.

    Returns:
        x_aug: Augmented tensor of shape (batch_size, 12, sequence_length).
        masks: Dictionary with 'mask_acc' and 'mask_gyro' shaped (batch_size,).
    """
    batch_size = x.size(0)
    device = x.device
    dtype = x.dtype

    if p_noise <= 0.0 and p_missing <= 0.0:
        # Fast path: no degradation applied
        mask_acc = torch.ones(batch_size, dtype=torch.float32, device=device)
        mask_gyro = torch.ones(batch_size, dtype=torch.float32, device=device)
        return x.clone(), {"mask_acc": mask_acc, "mask_gyro": mask_gyro}

    x_aug = x.clone()
    mask_acc = torch.ones(batch_size, dtype=torch.float32, device=device)
    mask_gyro = torch.ones(batch_size, dtype=torch.float32, device=device)

    # Random selection per sample
    rand_vals = torch.rand(batch_size, generator=generator)

    for i in range(batch_size):
        r = rand_vals[i].item()

        if r < p_missing:
            # Drop either ACC or GYRO randomly
            drop_acc = torch.rand(1, generator=generator).item() < 0.5
            if drop_acc:
                x_aug[i, ACC_INDICES, :] = 0.0
                mask_acc[i] = 0.0
            else:
                x_aug[i, GYRO_INDICES, :] = 0.0
                mask_gyro[i] = 0.0

        elif r < (p_missing + p_noise):
            # Add noise to either ACC, GYRO, or both
            noise_type = torch.randint(0, 3, (1,), generator=generator).item()
            if noise_type == 0:
                ch = ACC_INDICES
            elif noise_type == 1:
                ch = GYRO_INDICES
            else:
                ch = ACC_INDICES + GYRO_INDICES

            noise = torch.randn((len(ch), x.size(2)), generator=generator, device=device, dtype=dtype)
            x_aug[i, ch, :] = x_aug[i, ch, :] + (sigma * noise)

    return x_aug, {"mask_acc": mask_acc, "mask_gyro": mask_gyro}


# ============================================================================
# Self-Test Verification Routine
# ============================================================================
def run_self_test() -> None:
    """Comprehensive test assertions verifying all degradation requirements."""
    print("=" * 78)
    print("  RUNNING src/sensor_degradation.py COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 78)

    batch_size, channels, seq_len = 16, 12, 100
    x_original = torch.randn(batch_size, channels, seq_len)
    x_backup = x_original.clone()

    # ------------------------------------------------------------------------
    # Check 1: Non-Destructive Invariant
    # ------------------------------------------------------------------------
    print("\n[Check 1] Non-Destructive Invariant:")
    x_deg, _ = apply_condition(x_original, condition="noisy_both", sigma=1.0, seed=42)
    assert torch.equal(x_original, x_backup), "Input tensor was mutated in-place!"
    assert not torch.equal(x_deg, x_original), "Degraded output is identical to original!"
    print("  -> PASSED: Input tensor remains bit-identical after degradation.")

    # ------------------------------------------------------------------------
    # Check 2: Reproducibility with Seeded Generator
    # ------------------------------------------------------------------------
    print("\n[Check 2] Reproducibility with Seeded Generator:")
    deg_run1, _ = apply_condition(x_original, condition="noisy_acc", sigma=0.5, seed=42)
    deg_run2, _ = apply_condition(x_original, condition="noisy_acc", sigma=0.5, seed=42)
    deg_run3, _ = apply_condition(x_original, condition="noisy_acc", sigma=0.5, seed=999)

    assert torch.equal(deg_run1, deg_run2), "Same seed did not produce identical degradation!"
    assert not torch.equal(deg_run1, deg_run3), "Different seeds produced identical degradation!"
    print("  -> PASSED: Identical output guaranteed across runs with matching seed.")

    # ------------------------------------------------------------------------
    # Check 3: Missing Modality Isolation (missing_acc & missing_gyro)
    # ------------------------------------------------------------------------
    print("\n[Check 3] Missing Modality Isolation:")
    # Test missing_acc
    deg_miss_acc, masks_acc = apply_condition(x_original, condition="missing_acc")
    assert torch.all(masks_acc["mask_acc"] == 0.0), "mask_acc is not 0 for missing_acc!"
    assert torch.all(masks_acc["mask_gyro"] == 1.0), "mask_gyro is not 1 for missing_acc!"
    assert torch.all(deg_miss_acc[:, ACC_INDICES, :] == 0.0), "ACC channels not zeroed out!"
    assert torch.equal(deg_miss_acc[:, GYRO_INDICES, :], x_original[:, GYRO_INDICES, :]), (
        "GYRO channels were altered during missing_acc!"
    )
    print("  -> missing_acc zeroes exactly ACC channels and sets mask_acc=0.0.")

    # Test missing_gyro
    deg_miss_gyro, masks_gyro = apply_condition(x_original, condition="missing_gyro")
    assert torch.all(masks_gyro["mask_gyro"] == 0.0), "mask_gyro is not 0 for missing_gyro!"
    assert torch.all(masks_gyro["mask_acc"] == 1.0), "mask_acc is not 1 for missing_gyro!"
    assert torch.all(deg_miss_gyro[:, GYRO_INDICES, :] == 0.0), "GYRO channels not zeroed out!"
    assert torch.equal(deg_miss_gyro[:, ACC_INDICES, :], x_original[:, ACC_INDICES, :]), (
        "ACC channels were altered during missing_gyro!"
    )
    print("  -> missing_gyro zeroes exactly GYRO channels and sets mask_gyro=0.0.")
    print("  -> PASSED: Missing sensor channels perfectly zeroed with correct masks.")

    # ------------------------------------------------------------------------
    # Check 4: Noise Modality Isolation & Gaussian Variance
    # ------------------------------------------------------------------------
    print("\n[Check 4] Noise Modality Isolation & Variance (noisy_gyro):")
    target_sigma = 0.5
    deg_gyro, _ = apply_condition(x_original, condition="noisy_gyro", sigma=target_sigma, seed=42)

    # ACC channels must remain bit-identical
    assert torch.equal(deg_gyro[:, ACC_INDICES, :], x_original[:, ACC_INDICES, :]), (
        "ACC channels were modified during noisy_gyro!"
    )
    print("  -> ACC channels bit-identical during noisy_gyro.")

    # Measure noise on larger batch for statistical convergence
    x_large = torch.randn(128, 12, 200)
    deg_large, _ = apply_condition(x_large, condition="noisy_gyro", sigma=target_sigma, seed=42)
    noise_residual = deg_large[:, GYRO_INDICES, :] - x_large[:, GYRO_INDICES, :]
    measured_mean = noise_residual.mean().item()
    measured_std = noise_residual.std().item()

    print(f"  Target sigma:   {target_sigma:.4f}")
    print(f"  Measured mean:  {measured_mean:.4f} (Expected ≈ 0.0)")
    print(f"  Measured std:   {measured_std:.4f} (Expected ≈ {target_sigma:.4f})")

    assert abs(measured_mean) < 0.05, f"Noise mean not centered at 0: {measured_mean}"
    assert abs(measured_std - target_sigma) < 0.05, f"Noise std deviated from target: {measured_std}"
    print("  -> PASSED: Noise variance matches target sigma with zero mean.")

    # ------------------------------------------------------------------------
    # Check 5: Model Forward Wrapper (BaselineCNN vs AdaptiveSensorFusion)
    # ------------------------------------------------------------------------
    print("\n[Check 5] Forward Wrapper (BaselineCNN vs AdaptiveSensorFusion):")
    baseline = BaselineCNN()
    adaptive = AdaptiveSensorFusion()

    dummy_input = torch.randn(4, 12, 100)

    # 5a. Baseline forward with degradation
    logits_b = forward_with_condition(baseline, dummy_input, condition="noisy_acc", sigma=0.5)
    assert logits_b.shape == (4, 6)
    assert torch.isfinite(logits_b).all()
    print("  -> BaselineCNN forward_with_condition passed.")

    # 5b. Adaptive forward with degradation and dynamic weight extraction
    logits_a, weights = forward_with_condition(
        adaptive,
        dummy_input,
        condition="missing_acc",
        return_weights=True,
    )
    assert logits_a.shape == (4, 6)
    assert torch.isfinite(logits_a).all()
    w_acc_val = weights["w_acc"].mean().item()
    w_gyro_val = weights["w_gyro"].mean().item()
    print(f"  -> AdaptiveSensorFusion forward (missing_acc): w_acc={w_acc_val:.4f}, w_gyro={w_gyro_val:.4f}")
    assert w_acc_val == 0.0 and w_gyro_val == 1.0, "Adaptive model did not shift 100% to gyro branch!"
    print("  -> PASSED: Forward wrapper handles both architectures and propagates masks.")

    # ------------------------------------------------------------------------
    # Check 6: Degradation-Aware Training Augmentation
    # ------------------------------------------------------------------------
    print("\n[Check 6] Degradation-Aware Training Augmentation:")
    x_batch = torch.randn(32, 12, 100)
    x_aug, masks = degrade_batch_random(x_batch, p_noise=0.3, p_missing=0.3, sigma=0.5)

    assert x_aug.shape == x_batch.shape
    assert masks["mask_acc"].shape == (32,)
    assert masks["mask_gyro"].shape == (32,)
    # Verify non-destructive
    assert not torch.equal(x_aug, x_batch), "Augmented batch unchanged with positive probabilities!"
    print(f"  Augmented batch shape: {x_aug.shape} | Active Acc Masks mean: {masks['mask_acc'].mean().item():.2f}")
    print("  -> PASSED: Stochastic batch augmentation operational.")

    print("\n" + "=" * 78)
    print("  ALL SENSOR_DEGRADATION SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensor Degradation Engine (Member 2)")
    parser.add_argument("--test", action="store_true", help="Run comprehensive verification suite")
    parser.add_argument(
        "--condition",
        type=str,
        default="noisy_acc",
        choices=DEGRADATION_CONDITIONS,
        help="Degradation condition to inspect",
    )
    parser.add_argument("--sigma", type=float, default=None, help="Noise standard deviation override")
    args = parser.parse_args()

    # Always execute self-test
    run_self_test()


if __name__ == "__main__":
    main()
