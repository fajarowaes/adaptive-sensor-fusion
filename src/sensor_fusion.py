"""
src/sensor_fusion.py

Adaptive Sensor Fusion Architecture for Human Activity Recognition (HAR)
using the MotionSense dataset.

This module implements a dual-branch neural network with dynamic, reliability-based
sensor fusion (AdaptiveSensorFusion). The architecture processes accelerometer-related
features and gyroscope/attitude features in separate specialized branches, evaluates
their relative reliability and availability, and dynamically calculates adaptive fusion
weights to remain robust against sensor failure, corruption, or missing modalities.

Branch Partitioning (consistent with MotionSense dataset):
    - Branch 1 (Accelerometer & Gravity - 6 channels):
        gravity.x, gravity.y, gravity.z,
        userAcceleration.x, userAcceleration.y, userAcceleration.z
    - Branch 2 (Gyroscope & Attitude - 6 channels):
        attitude.roll, attitude.pitch, attitude.yaw,
        rotationRate.x, rotationRate.y, rotationRate.z

Target Activities (6 classes):
    0: downstairs, 1: upstairs, 2: walking,
    3: jogging, 4: standing, 5: sitting
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Dataset Feature Grouping Indices
# ============================================================================

# Full 12-feature order from Phase 2 preprocessing
SENSOR_COLUMNS = [
    "attitude.roll",       # idx 0 -> Branch 2 (Gyro/Attitude)
    "attitude.pitch",      # idx 1 -> Branch 2 (Gyro/Attitude)
    "attitude.yaw",        # idx 2 -> Branch 2 (Gyro/Attitude)
    "gravity.x",           # idx 3 -> Branch 1 (Accel/Gravity)
    "gravity.y",           # idx 4 -> Branch 1 (Accel/Gravity)
    "gravity.z",           # idx 5 -> Branch 1 (Accel/Gravity)
    "rotationRate.x",      # idx 6 -> Branch 2 (Gyro/Attitude)
    "rotationRate.y",      # idx 7 -> Branch 2 (Gyro/Attitude)
    "rotationRate.z",      # idx 8 -> Branch 2 (Gyro/Attitude)
    "userAcceleration.x",  # idx 9 -> Branch 1 (Accel/Gravity)
    "userAcceleration.y",  # idx 10 -> Branch 1 (Accel/Gravity)
    "userAcceleration.z",  # idx 11 -> Branch 1 (Accel/Gravity)
]

# Indices within the 12-feature tensor
ACC_INDICES = [3, 4, 5, 9, 10, 11]    # gravity (3-5) + userAcceleration (9-11)
GYRO_INDICES = [0, 1, 2, 6, 7, 8]     # attitude (0-2) + rotationRate (6-8)

ACC_CHANNEL_NAMES = [SENSOR_COLUMNS[i] for i in ACC_INDICES]
GYRO_CHANNEL_NAMES = [SENSOR_COLUMNS[i] for i in GYRO_INDICES]

NUM_BRANCH_CHANNELS = 6
NUM_ACTIVITIES = 6


# ============================================================================
# Dedicated Sensor Branch Feature Extractor
# ============================================================================

class SensorBranchEncoder(nn.Module):
    """
    Dedicated 1D CNN feature extractor for a specific sensor modality group.

    Architecture:
        - Block 1: Conv1D(6 -> 64, kernel=5, pad=2) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
        - Block 2: Conv1D(64 -> 128, kernel=5, pad=2) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
        - Block 3: Conv1D(128 -> feature_dim, kernel=3, pad=1) + BatchNorm + ReLU + AdaptiveAvgPool1D(1)
        - Output Projection: Linear(feature_dim -> feature_dim) + LayerNorm + ReLU

    Input:
        (batch_size, 6, sequence_length)
    Output:
        Latent embedding vector of shape (batch_size, feature_dim).
    """

    def __init__(
        self,
        in_channels: int = NUM_BRANCH_CHANNELS,
        feature_dim: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate),
            # Block 3
            nn.Conv1d(128, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Dropout(p=dropout_rate),
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal representation for one sensor branch.

        Args:
            x: Tensor of shape (batch_size, 6, sequence_length).

        Returns:
            Latent representation tensor of shape (batch_size, feature_dim).
        """
        feat = self.conv_blocks(x)        # -> (batch_size, feature_dim, 1)
        feat = feat.squeeze(-1)           # -> (batch_size, feature_dim)
        embedding = self.projector(feat)  # -> (batch_size, feature_dim)
        return embedding


# ============================================================================
# Adaptive Fusion Mechanism with Dynamic Reliability Weighting
# ============================================================================

class AdaptiveFusionModule(nn.Module):
    """
    Adaptive sensor fusion mechanism with reliability gating and availability masking.

    Mathematical Formulation:
        Let h_acc, h_gyro in R^{B x D} be the latent embeddings of the accelerometer
        and gyroscope branches, respectively.

        1. Sensor Availability Masks:
           m_acc, m_gyro in {0, 1} indicate whether the physical sensor stream is active.
           If a sensor stream is completely unavailable, m_i = 0.

        2. Sensor Reliability Indicators:
           r_acc, r_gyro in [0, 1] represent signal quality / reliability metrics
           (e.g., from SNR estimation, sensor variance, or simulated noise degradation).
           When not provided externally, a learned self-confidence gating network
           estimates reliability from the branch representations.

        3. Effective Reliability:
           R_acc  = m_acc  * r_acc
           R_gyro = m_gyro * r_gyro

        4. Content-based Context Scores:
           s_acc  = G_acc(h_acc)   in R^{B x 1}
           s_gyro = G_gyro(h_gyro) in R^{B x 1}

        5. Dynamic Weight Calculation with Strict Zero-Division Prevention:
           alpha_acc  = R_acc  * exp(s_acc)
           alpha_gyro = R_gyro * exp(s_gyro)
           denominator = alpha_acc + alpha_gyro + epsilon (where epsilon = 1e-8)

           w_acc  = alpha_acc / denominator
           w_gyro = alpha_gyro / denominator

        6. Boundary Guarantee:
           - If m_acc == 0 and m_gyro > 0:  w_acc = 0.0, w_gyro = 1.0 (100% Gyro reliance)
           - If m_gyro == 0 and m_acc > 0:  w_gyro = 0.0, w_acc = 1.0 (100% Accel reliance)
           - If both available: weights adapt proportionally to reliability and signal content.
           - If both unavailable: graceful fallback w_acc = 0.5, w_gyro = 0.5 (neutral fallback).

        7. Masked Latent Modulation:
           h_acc_eff  = h_acc  * m_acc
           h_gyro_eff = h_gyro * m_gyro
           h_fused = [w_acc * h_acc_eff,  w_gyro * h_gyro_eff]
    """

    def __init__(self, feature_dim: int = 128, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon

        # Content-based relevance / importance scoring network
        self.importance_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
        )

        # Internal learned confidence estimator (used if external reliability is None)
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid(),
        )

    def _format_indicator(
        self,
        indicator: Optional[Union[float, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        default_value: float = 1.0,
    ) -> torch.Tensor:
        """Standardizes scalar, list, or tensor indicators into shape (batch_size, 1)."""
        if indicator is None:
            return torch.full((batch_size, 1), default_value, device=device, dtype=torch.float32)

        if isinstance(indicator, (int, float)):
            return torch.full((batch_size, 1), float(indicator), device=device, dtype=torch.float32)

        if not isinstance(indicator, torch.Tensor):
            indicator = torch.tensor(indicator, device=device, dtype=torch.float32)
        else:
            indicator = indicator.to(device=device, dtype=torch.float32)

        if indicator.dim() == 0:
            indicator = indicator.view(1, 1).expand(batch_size, 1)
        elif indicator.dim() == 1:
            if indicator.size(0) == 1:
                indicator = indicator.view(1, 1).expand(batch_size, 1)
            else:
                indicator = indicator.view(-1, 1)

        return indicator

    def forward(
        self,
        h_acc: torch.Tensor,
        h_gyro: torch.Tensor,
        mask_acc: Optional[Union[float, torch.Tensor]] = None,
        mask_gyro: Optional[Union[float, torch.Tensor]] = None,
        reliability_acc: Optional[Union[float, torch.Tensor]] = None,
        reliability_gyro: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes reliability-weighted adaptive sensor fusion.

        Args:
            h_acc: Accelerometer branch latent representation (batch_size, feature_dim).
            h_gyro: Gyroscope branch latent representation (batch_size, feature_dim).
            mask_acc: Availability mask for accelerometer (1=available, 0=unavailable).
            mask_gyro: Availability mask for gyroscope (1=available, 0=unavailable).
            reliability_acc: Quality/reliability score for accelerometer in [0, 1].
            reliability_gyro: Quality/reliability score for gyroscope in [0, 1].

        Returns:
            fused_representation: Fused feature tensor of shape (batch_size, feature_dim * 2).
            weights_info: Dictionary containing 'w_acc', 'w_gyro', 'r_acc', 'r_gyro'.
        """
        batch_size = h_acc.size(0)
        device = h_acc.device

        # Standardize availability masks (0.0 or 1.0)
        m_acc = self._format_indicator(mask_acc, batch_size, device, default_value=1.0)
        m_gyro = self._format_indicator(mask_gyro, batch_size, device, default_value=1.0)

        # Ensure masks are clamped between 0 and 1
        m_acc = torch.clamp(m_acc, 0.0, 1.0)
        m_gyro = torch.clamp(m_gyro, 0.0, 1.0)

        # Standardize reliability indicators (if None, use learned self-confidence)
        if reliability_acc is None:
            r_acc = self.confidence_estimator(h_acc)
        else:
            r_acc = self._format_indicator(reliability_acc, batch_size, device, default_value=1.0)

        if reliability_gyro is None:
            r_gyro = self.confidence_estimator(h_gyro)
        else:
            r_gyro = self._format_indicator(reliability_gyro, batch_size, device, default_value=1.0)

        # Clamp reliability values to [0.0, 1.0] to guarantee numerical stability
        r_acc = torch.clamp(r_acc, 0.0, 1.0)
        r_gyro = torch.clamp(r_gyro, 0.0, 1.0)

        # Effective reliability incorporates availability gating
        eff_rel_acc = m_acc * r_acc
        eff_rel_gyro = m_gyro * r_gyro

        # Content-based importance scoring
        score_acc = self.importance_gate(h_acc)      # -> (batch_size, 1)
        score_gyro = self.importance_gate(h_gyro)    # -> (batch_size, 1)

        # Prevent exponent overflow by shifting by max score
        max_score = torch.max(score_acc, score_gyro)
        exp_acc = torch.exp(score_acc - max_score)
        exp_gyro = torch.exp(score_gyro - max_score)

        # Unnormalized importance weighted by effective reliability
        alpha_acc = eff_rel_acc * exp_acc
        alpha_gyro = eff_rel_gyro * exp_gyro

        # Denominator with epsilon to prevent division by zero
        denom = alpha_acc + alpha_gyro + self.epsilon

        w_acc = alpha_acc / denom
        w_gyro = alpha_gyro / denom

        # --------------------------------------------------------------------
        # Boundary Conditions Handling:
        # Guarantee exact 0.0 / 1.0 boundary behavior when one branch is masked out
        # --------------------------------------------------------------------
        acc_missing = (m_acc < 0.5) & (m_gyro >= 0.5)
        gyro_missing = (m_gyro < 0.5) & (m_acc >= 0.5)
        both_missing = (m_acc < 0.5) & (m_gyro < 0.5)

        w_acc = torch.where(acc_missing, torch.zeros_like(w_acc), w_acc)
        w_gyro = torch.where(acc_missing, torch.ones_like(w_gyro), w_gyro)

        w_gyro = torch.where(gyro_missing, torch.zeros_like(w_gyro), w_gyro)
        w_acc = torch.where(gyro_missing, torch.ones_like(w_acc), w_acc)

        # Fallback if both sensors fail completely
        w_acc = torch.where(both_missing, torch.full_like(w_acc, 0.5), w_acc)
        w_gyro = torch.where(both_missing, torch.full_like(w_gyro, 0.5), w_gyro)

        # Zero out feature representation of missing sensor to prevent corrupted signal leakage
        h_acc_eff = h_acc * m_acc
        h_gyro_eff = h_gyro * m_gyro

        # Weighted modulation of latent representations
        h_acc_weighted = w_acc * h_acc_eff
        h_gyro_weighted = w_gyro * h_gyro_eff

        # Dual-branch fusion: concatenation preserves distinct modal semantics,
        # while adaptive weights dynamically suppress unreliable channels.
        fused_representation = torch.cat([h_acc_weighted, h_gyro_weighted], dim=-1)

        weights_info = {
            "w_acc": w_acc,
            "w_gyro": w_gyro,
            "eff_rel_acc": eff_rel_acc,
            "eff_rel_gyro": eff_rel_gyro,
            "mask_acc": m_acc,
            "mask_gyro": m_gyro,
        }

        return fused_representation, weights_info


# ============================================================================
# Complete Adaptive Sensor Fusion Model
# ============================================================================

class AdaptiveSensorFusion(nn.Module):
    """
    Two-branch Adaptive Sensor Fusion model for Human Activity Recognition.

    Processes 12 MotionSense sensor features partitioned into two domain streams:
        1. Accelerometer & Gravity (6 features)
        2. Gyroscope & Attitude (6 features)

    Inputs:
        Flexible interface supporting either:
        - Unified tensor: x of shape (batch_size, 12, sequence_length)
          Columns are automatically partitioned into Branch 1 and Branch 2.
        - Separate branch tensors: x_acc (batch_size, 6, sequence_length) and
          x_gyro (batch_size, 6, sequence_length).

    Reliability & Availability Controls:
        - mask_acc, mask_gyro: Availability masks (1.0 = available, 0.0 = missing).
        - reliability_acc, reliability_gyro: Signal quality metrics in [0.0, 1.0].

    Output:
        Logits of shape (batch_size, num_classes=6) for CrossEntropyLoss.
        If return_weights=True, returns (logits, weights_dict).
    """

    def __init__(
        self,
        in_channels_acc: int = NUM_BRANCH_CHANNELS,
        in_channels_gyro: int = NUM_BRANCH_CHANNELS,
        branch_feature_dim: int = 128,
        num_classes: int = NUM_ACTIVITIES,
        dropout_rate: float = 0.2,
        dense_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.branch_feature_dim = branch_feature_dim
        self.num_classes = num_classes

        # Specialized branch encoders
        self.acc_branch = SensorBranchEncoder(
            in_channels=in_channels_acc,
            feature_dim=branch_feature_dim,
            dropout_rate=dropout_rate,
        )

        self.gyro_branch = SensorBranchEncoder(
            in_channels=in_channels_gyro,
            feature_dim=branch_feature_dim,
            dropout_rate=dropout_rate,
        )

        # Adaptive fusion mechanism
        self.fusion_module = AdaptiveFusionModule(
            feature_dim=branch_feature_dim,
            epsilon=1e-8,
        )

        # Classification Head (processes fused 2 * branch_feature_dim representation)
        fused_dim = branch_feature_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_classifier_weights()

    def _initialize_classifier_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _split_unified_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits a 12-channel sensor tensor into accelerometer and gyroscope branches.

        Uses ACC_INDICES and GYRO_INDICES defined from the MotionSense feature order.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch, channels, length), but got {x.dim()}D tensor of shape {x.shape}"
            )

        # Handle (batch_size, sequence_length, channels)
        if x.size(1) != 12 and x.size(2) == 12:
            x = x.transpose(1, 2)

        if x.size(1) != 12:
            raise ValueError(
                f"Expected 12 sensor channels for unified input, but got {x.size(1)} channels (shape: {x.shape})"
            )

        device = x.device
        acc_idx = torch.tensor(ACC_INDICES, dtype=torch.long, device=device)
        gyro_idx = torch.tensor(GYRO_INDICES, dtype=torch.long, device=device)

        x_acc = torch.index_select(x, dim=1, index=acc_idx)
        x_gyro = torch.index_select(x, dim=1, index=gyro_idx)
        return x_acc, x_gyro

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        x_acc: Optional[torch.Tensor] = None,
        x_gyro: Optional[torch.Tensor] = None,
        mask_acc: Optional[Union[float, torch.Tensor]] = None,
        mask_gyro: Optional[Union[float, torch.Tensor]] = None,
        reliability_acc: Optional[Union[float, torch.Tensor]] = None,
        reliability_gyro: Optional[Union[float, torch.Tensor]] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for adaptive sensor fusion.

        Args:
            x: Unified tensor of shape (batch_size, 12, sequence_length).
            x_acc: Accelerometer tensor of shape (batch_size, 6, sequence_length).
            x_gyro: Gyroscope tensor of shape (batch_size, 6, sequence_length).
            mask_acc: Availability mask for accelerometer (1=available, 0=unavailable).
            mask_gyro: Availability mask for gyroscope (1=available, 0=unavailable).
            reliability_acc: Quality/reliability score for accelerometer in [0, 1].
            reliability_gyro: Quality/reliability score for gyroscope in [0, 1].
            return_weights: If True, returns (logits, weights_info).

        Returns:
            Logits of shape (batch_size, 6).
            If return_weights=True, returns (logits, weights_info).
        """
        # Resolve inputs
        if x is not None:
            branch_acc, branch_gyro = self._split_unified_input(x)
        elif x_acc is not None and x_gyro is not None:
            # Handle potential (batch_size, length, 6) orientation
            if x_acc.size(1) != NUM_BRANCH_CHANNELS and x_acc.size(2) == NUM_BRANCH_CHANNELS:
                x_acc = x_acc.transpose(1, 2)
            if x_gyro.size(1) != NUM_BRANCH_CHANNELS and x_gyro.size(2) == NUM_BRANCH_CHANNELS:
                x_gyro = x_gyro.transpose(1, 2)
            branch_acc, branch_gyro = x_acc, x_gyro
        else:
            raise ValueError(
                "Either unified 'x' (12 channels) or both 'x_acc' and 'x_gyro' (6 channels each) must be provided."
            )

        # Extract features through specialized branch encoders
        h_acc = self.acc_branch(branch_acc)      # -> (batch_size, branch_feature_dim)
        h_gyro = self.gyro_branch(branch_gyro)    # -> (batch_size, branch_feature_dim)

        # Dynamically compute fusion weights and fused representation
        fused_h, weights_info = self.fusion_module(
            h_acc=h_acc,
            h_gyro=h_gyro,
            mask_acc=mask_acc,
            mask_gyro=mask_gyro,
            reliability_acc=reliability_acc,
            reliability_gyro=reliability_gyro,
        )

        # Classification logits
        logits = self.classifier(fused_h)         # -> (batch_size, 6)

        if return_weights:
            return logits, weights_info
        return logits

    @torch.no_grad()
    def predict(
        self,
        x: Optional[torch.Tensor] = None,
        x_acc: Optional[torch.Tensor] = None,
        x_gyro: Optional[torch.Tensor] = None,
        mask_acc: Optional[Union[float, torch.Tensor]] = None,
        mask_gyro: Optional[Union[float, torch.Tensor]] = None,
        reliability_acc: Optional[Union[float, torch.Tensor]] = None,
        reliability_gyro: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Inference interface: returns predicted class indices (0-5).
        """
        self.eval()
        logits = self.forward(
            x=x,
            x_acc=x_acc,
            x_gyro=x_gyro,
            mask_acc=mask_acc,
            mask_gyro=mask_gyro,
            reliability_acc=reliability_acc,
            reliability_gyro=reliability_gyro,
            return_weights=False,
        )
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(
        self,
        x: Optional[torch.Tensor] = None,
        x_acc: Optional[torch.Tensor] = None,
        x_gyro: Optional[torch.Tensor] = None,
        mask_acc: Optional[Union[float, torch.Tensor]] = None,
        mask_gyro: Optional[Union[float, torch.Tensor]] = None,
        reliability_acc: Optional[Union[float, torch.Tensor]] = None,
        reliability_gyro: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Inference interface: returns normalized class probabilities via softmax.
        """
        self.eval()
        logits = self.forward(
            x=x,
            x_acc=x_acc,
            x_gyro=x_gyro,
            mask_acc=mask_acc,
            mask_gyro=mask_gyro,
            reliability_acc=reliability_acc,
            reliability_gyro=reliability_gyro,
            return_weights=False,
        )
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts for model and sub-modules."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        acc_params = sum(p.numel() for p in self.acc_branch.parameters())
        gyro_params = sum(p.numel() for p in self.gyro_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "acc_branch_parameters": acc_params,
            "gyro_branch_parameters": gyro_params,
            "fusion_module_parameters": fusion_params,
            "classifier_parameters": classifier_params,
        }

    def get_loss_criterion(self) -> nn.Module:
        """Returns CrossEntropyLoss for classification."""
        return nn.CrossEntropyLoss()

    def get_optimizer(
        self, learning_rate: float = 1e-3, weight_decay: float = 1e-4
    ) -> torch.optim.Optimizer:
        """Returns the recommended Adam optimizer for Member 2's training pipeline."""
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )


# ============================================================================
# Checkpoint Serialization Helpers
# ============================================================================

def save_adaptive_model(model: nn.Module, filepath: str) -> None:
    """Saves model state dictionary to disk."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        },
        filepath,
    )


def load_adaptive_model(
    model: nn.Module, filepath: str, map_location: Optional[str] = None
) -> nn.Module:
    """Loads state dictionary into an AdaptiveSensorFusion instance."""
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ============================================================================
# Comprehensive Model Testing Suite
# ============================================================================

def test_adaptive_sensor_fusion() -> None:
    """
    Comprehensive verification suite for AdaptiveSensorFusion:
        1. Normal condition (both sensors available, default reliability).
        2. Accelerometer missing (mask_acc=0, mask_gyro=1).
        3. Gyroscope missing (mask_acc=1, mask_gyro=0).
        4. Unequal sensor reliability (e.g. noisy accelerometer vs. clean gyroscope).
        5. Separate branch input mode (x_acc, x_gyro).
        6. Numerical stability & edge cases (both missing, all zeros, variable window lengths).
    """
    print("=" * 78)
    print("  TESTING ADAPTIVE SENSOR FUSION MODEL (src/sensor_fusion.py)")
    print("=" * 78)

    model = AdaptiveSensorFusion(
        in_channels_acc=6,
        in_channels_gyro=6,
        branch_feature_dim=128,
        num_classes=6,
    )
    counts = model.count_parameters()
    print(f"Model Architecture:          {model.__class__.__name__}")
    print(f"Total Parameters:            {counts['total_parameters']:,}")
    print(f"  - Acc Branch Encoder:      {counts['acc_branch_parameters']:,}")
    print(f"  - Gyro Branch Encoder:     {counts['gyro_branch_parameters']:,}")
    print(f"  - Adaptive Fusion Module:  {counts['fusion_module_parameters']:,}")
    print(f"  - Classification Head:     {counts['classifier_parameters']:,}")
    print("-" * 78)

    batch_size = 8
    seq_len = 128
    dummy_12ch = torch.randn(batch_size, 12, seq_len)

    # ------------------------------------------------------------------------
    # Test 1: Normal condition (Both sensors fully available)
    # ------------------------------------------------------------------------
    logits_norm, w_info = model(
        x=dummy_12ch,
        mask_acc=1.0,
        mask_gyro=1.0,
        return_weights=True,
    )
    assert logits_norm.shape == (batch_size, 6), f"Shape error: {logits_norm.shape}"
    assert torch.isfinite(logits_norm).all(), "Logits contain non-finite values!"
    w_acc_mean = w_info["w_acc"].mean().item()
    w_gyro_mean = w_info["w_gyro"].mean().item()
    print(f"Test 1 [Normal: Both Available]:")
    print(f"  Output Shape: {logits_norm.shape} | All Finite: {torch.isfinite(logits_norm).all().item()}")
    print(f"  Mean Weights: w_acc={w_acc_mean:.4f}, w_gyro={w_gyro_mean:.4f} (Sum: {w_acc_mean + w_gyro_mean:.4f})")
    assert abs((w_acc_mean + w_gyro_mean) - 1.0) < 1e-4, "Weights must sum to 1.0"
    print("  -> PASSED: Both branches actively contribute.\n")

    # ------------------------------------------------------------------------
    # Test 2: Accelerometer Missing (mask_acc=0, mask_gyro=1)
    # ------------------------------------------------------------------------
    logits_no_acc, w_info_no_acc = model(
        x=dummy_12ch,
        mask_acc=0.0,
        mask_gyro=1.0,
        return_weights=True,
    )
    assert logits_no_acc.shape == (batch_size, 6)
    assert torch.isfinite(logits_no_acc).all()
    w_acc_zero = w_info_no_acc["w_acc"]
    w_gyro_one = w_info_no_acc["w_gyro"]
    assert torch.allclose(w_acc_zero, torch.zeros_like(w_acc_zero)), "w_acc must be 0 when acc is missing!"
    assert torch.allclose(w_gyro_one, torch.ones_like(w_gyro_one)), "w_gyro must be 1 when acc is missing!"
    print(f"Test 2 [Accelerometer Missing]:")
    print(f"  Output Shape: {logits_no_acc.shape} | All Finite: {torch.isfinite(logits_no_acc).all().item()}")
    print(f"  Weights:      w_acc={w_acc_zero[0].item():.4f}, w_gyro={w_gyro_one[0].item():.4f}")
    print("  -> PASSED: Model shifts 100% reliance to Gyroscope/Attitude branch.\n")

    # ------------------------------------------------------------------------
    # Test 3: Gyroscope/Attitude Missing (mask_acc=1, mask_gyro=0)
    # ------------------------------------------------------------------------
    logits_no_gyro, w_info_no_gyro = model(
        x=dummy_12ch,
        mask_acc=1.0,
        mask_gyro=0.0,
        return_weights=True,
    )
    assert logits_no_gyro.shape == (batch_size, 6)
    assert torch.isfinite(logits_no_gyro).all()
    w_acc_one = w_info_no_gyro["w_acc"]
    w_gyro_zero = w_info_no_gyro["w_gyro"]
    assert torch.allclose(w_acc_one, torch.ones_like(w_acc_one)), "w_acc must be 1 when gyro is missing!"
    assert torch.allclose(w_gyro_zero, torch.zeros_like(w_gyro_zero)), "w_gyro must be 0 when gyro is missing!"
    print(f"Test 3 [Gyroscope Missing]:")
    print(f"  Output Shape: {logits_no_gyro.shape} | All Finite: {torch.isfinite(logits_no_gyro).all().item()}")
    print(f"  Weights:      w_acc={w_acc_one[0].item():.4f}, w_gyro={w_gyro_zero[0].item():.4f}")
    print("  -> PASSED: Model shifts 100% reliance to Accelerometer/Gravity branch.\n")

    # ------------------------------------------------------------------------
    # Test 4: Unequal Reliability (e.g. Noisy Accel vs. Clean Gyro)
    # ------------------------------------------------------------------------
    # Scenario A: Accel degraded (r_acc=0.15), Gyro pristine (r_gyro=0.95)
    _, w_degraded_acc = model(
        x=dummy_12ch,
        mask_acc=1.0,
        mask_gyro=1.0,
        reliability_acc=0.15,
        reliability_gyro=0.95,
        return_weights=True,
    )
    w_a_acc = w_degraded_acc["w_acc"].mean().item()
    w_a_gyro = w_degraded_acc["w_gyro"].mean().item()

    # Scenario B: Gyro degraded (r_gyro=0.15), Accel pristine (r_acc=0.95)
    _, w_degraded_gyro = model(
        x=dummy_12ch,
        mask_acc=1.0,
        mask_gyro=1.0,
        reliability_acc=0.95,
        reliability_gyro=0.15,
        return_weights=True,
    )
    w_b_acc = w_degraded_gyro["w_acc"].mean().item()
    w_b_gyro = w_degraded_gyro["w_gyro"].mean().item()

    print(f"Test 4 [Dynamic Reliability Weighting]:")
    print(f"  Scenario A (r_acc=0.15, r_gyro=0.95) -> w_acc={w_a_acc:.4f}, w_gyro={w_a_gyro:.4f}")
    print(f"  Scenario B (r_acc=0.95, r_gyro=0.15) -> w_acc={w_b_acc:.4f}, w_gyro={w_b_gyro:.4f}")
    assert w_a_acc < w_a_gyro, "Weights did not shift to more reliable gyro branch!"
    assert w_b_acc > w_b_gyro, "Weights did not shift to more reliable accel branch!"
    print("  -> PASSED: Fusion weights dynamically favor the higher-reliability sensor.\n")

    # ------------------------------------------------------------------------
    # Test 5: Separate Branch Input Interface (x_acc and x_gyro)
    # ------------------------------------------------------------------------
    dummy_acc = torch.randn(batch_size, 6, seq_len)
    dummy_gyro = torch.randn(batch_size, 6, seq_len)
    logits_sep = model(x_acc=dummy_acc, x_gyro=dummy_gyro)
    assert logits_sep.shape == (batch_size, 6)
    assert torch.isfinite(logits_sep).all()
    print("Test 5 [Separate Branch Input Interface]:")
    print(f"  Output Shape: {logits_sep.shape} | All Finite: {torch.isfinite(logits_sep).all().item()}")
    print("  -> PASSED: Dedicated (x_acc, x_gyro) forward pass works seamlessly.\n")

    # ------------------------------------------------------------------------
    # Test 6: Numerical Stability & Edge Cases
    # ------------------------------------------------------------------------
    # Edge case 6a: Both sensors missing (mask_acc=0, mask_gyro=0)
    logits_both_missing, w_both = model(
        x=dummy_12ch,
        mask_acc=0.0,
        mask_gyro=0.0,
        return_weights=True,
    )
    assert logits_both_missing.shape == (batch_size, 6)
    assert torch.isfinite(logits_both_missing).all(), "NaN/Inf when both sensors missing!"
    print("Test 6a [Both Sensors Missing Fallback]:")
    print(f"  Output Shape: {logits_both_missing.shape} | All Finite: {torch.isfinite(logits_both_missing).all().item()}")
    print(f"  Weights: w_acc={w_both['w_acc'][0].item():.2f}, w_gyro={w_both['w_gyro'][0].item():.2f}")
    print("  -> PASSED: Division-by-zero safely avoided, finite output guaranteed.")

    # Edge case 6b: All-zero sensor values with extreme zero reliability
    zero_input = torch.zeros(batch_size, 12, seq_len)
    logits_zeros = model(x=zero_input, reliability_acc=0.0, reliability_gyro=0.0)
    assert torch.isfinite(logits_zeros).all()
    print("Test 6b [All-Zero Input with 0.0 Reliability]:")
    print(f"  Output Shape: {logits_zeros.shape} | All Finite: {torch.isfinite(logits_zeros).all().item()}")
    print("  -> PASSED: Extreme zero input does not cause NaN or Inf.\n")

    # ------------------------------------------------------------------------
    # Test 7: Inference Methods and Variable Sequence Lengths
    # ------------------------------------------------------------------------
    preds = model.predict(dummy_12ch)
    probs = model.predict_proba(dummy_12ch)
    assert preds.shape == (batch_size,)
    assert probs.shape == (batch_size, 6)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    criterion = model.get_loss_criterion()
    optimizer = model.get_optimizer(learning_rate=1e-3)
    loss = criterion(logits_norm, torch.randint(0, 6, (batch_size,)))
    assert torch.isfinite(loss).item()

    for length in [64, 200, 256]:
        var_out = model(x=torch.randn(2, 12, length))
        assert var_out.shape == (2, 6)
        assert torch.isfinite(var_out).all()
    print("Test 7 [Inference, Loss, Optimizer, and Variable Lengths]:")
    print(f"  Predict shape: {preds.shape} | Probs sum: {probs.sum(dim=1).mean().item():.4f}")
    print(f"  Criterion: {criterion.__class__.__name__} (Loss={loss.item():.4f}) | Optimizer: {optimizer.__class__.__name__}")
    print("  -> PASSED: Inference and variable window lengths verified.")

    print("=" * 78)
    print("  ALL ADAPTIVE SENSOR FUSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    test_adaptive_sensor_fusion()

