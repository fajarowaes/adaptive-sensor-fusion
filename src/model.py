"""
src/model.py

Baseline 1D Convolutional Neural Network (CNN) for Human Activity Recognition (HAR)
using the MotionSense dataset.

This module implements the single-stream baseline architecture (BaselineCNN) that
processes all 12 smartphone sensor features simultaneously over temporal windows.

Sensor Feature Order (matches MotionSense preprocessed dataset):
    0: attitude.roll
    1: attitude.pitch
    2: attitude.yaw
    3: gravity.x
    4: gravity.y
    5: gravity.z
    6: rotationRate.x
    7: rotationRate.y
    8: rotationRate.z
    9: userAcceleration.x
    10: userAcceleration.y
    11: userAcceleration.z

Target Activities (6 classes):
    0: downstairs
    1: upstairs
    2: walking
    3: jogging
    4: standing
    5: sitting
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Dataset Constants and Feature Order
# ============================================================================

SENSOR_COLUMNS = [
    "attitude.roll",
    "attitude.pitch",
    "attitude.yaw",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
]

NUM_SENSOR_FEATURES = 12
NUM_ACTIVITIES = 6

ACTIVITY_CLASSES = [
    "downstairs",
    "upstairs",
    "walking",
    "jogging",
    "standing",
    "sitting",
]


# ============================================================================
# Baseline 1D CNN Architecture
# ============================================================================

class BaselineCNN(nn.Module):
    """
    1D Convolutional Neural Network baseline model for Human Activity Recognition.

    Processes all 12 motion sensor features over a time-series window.
    Architecture:
        - Conv1D Block 1: Conv1D(12 -> 64, kernel=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
        - Conv1D Block 2: Conv1D(64 -> 128, kernel=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
        - Conv1D Block 3: Conv1D(128 -> 256, kernel=3) + BatchNorm + ReLU + AdaptiveAvgPool1D(1) + Dropout(0.3)
        - Classifier Head: Linear(256 -> 128) + ReLU + Dropout(0.3) + Linear(128 -> 6)

    Expected Input Shape:
        Standard PyTorch format: (batch_size, in_channels=12, sequence_length)
        If provided as (batch_size, sequence_length, 12), the forward method
        automatically permutes to (batch_size, 12, sequence_length).

    Output Shape:
        (batch_size, num_classes=6) unnormalized logits for CrossEntropyLoss.
    """

    def __init__(
        self,
        in_channels: int = NUM_SENSOR_FEATURES,
        num_classes: int = NUM_ACTIVITIES,
        dropout_rate: float = 0.2,
        dense_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Temporal Feature Extraction Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate),
        )

        # Temporal Feature Extraction Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate),
        )

        # Temporal Feature Extraction Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Dropout(p=dense_dropout),
        )

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the baseline CNN.

        Args:
            x: Input tensor of shape (batch_size, 12, sequence_length)
               or (batch_size, sequence_length, 12).

        Returns:
            Logits of shape (batch_size, num_classes=6).
        """
        # Automatically handle (batch_size, sequence_length, channels) format
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch, channels, length), but got {x.dim()}D tensor of shape {x.shape}"
            )
        if x.size(1) != self.in_channels and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.size(1)} channels (shape: {x.shape})"
            )

        # Feature extraction
        out = self.block1(x)       # -> (batch_size, 64, L/2)
        out = self.block2(out)     # -> (batch_size, 128, L/4)
        out = self.block3(out)     # -> (batch_size, 256, 1)

        # Flatten spatial/temporal dimension
        out = out.squeeze(-1)      # -> (batch_size, 256)

        # Classification logits
        logits = self.classifier(out)  # -> (batch_size, 6)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference interface: returns predicted class indices.

        Args:
            x: Input tensor of shape (batch_size, 12, sequence_length).

        Returns:
            torch.Tensor of shape (batch_size,) with integer class predictions (0-5).
        """
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference interface: returns normalized activity probabilities via softmax.

        Args:
            x: Input tensor of shape (batch_size, 12, sequence_length).

        Returns:
            torch.Tensor of shape (batch_size, 6) with probabilities summing to 1.
        """
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def get_loss_criterion(self) -> nn.Module:
        """Returns the recommended classification loss function (CrossEntropyLoss)."""
        return nn.CrossEntropyLoss()

    def get_optimizer(
        self, learning_rate: float = 1e-3, weight_decay: float = 1e-4
    ) -> torch.optim.Optimizer:
        """Returns the recommended Adam optimizer for Member 2's training pipeline."""
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )


# ============================================================================
# Checkpoint Serialization Helpers (for Member 2 and Member 3)
# ============================================================================

def save_model(model: nn.Module, filepath: str) -> None:
    """
    Saves model state dictionary to disk.

    Args:
        model: PyTorch model instance.
        filepath: Destination file path (e.g. 'models/baseline_cnn.pth').
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        },
        filepath,
    )


def load_model(
    model: nn.Module, filepath: str, map_location: Optional[str] = None
) -> nn.Module:
    """
    Loads saved state dictionary into model instance.

    Args:
        model: PyTorch model instance with matching architecture.
        filepath: Checkpoint file path.
        map_location: Device mapping (e.g. 'cpu' or 'cuda').

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ============================================================================
# Verification & Self-Test Routine
# ============================================================================

def test_baseline_cnn() -> None:
    """Comprehensive test routine for BaselineCNN model."""
    print("=" * 70)
    print("  TESTING BASELINE 1D CNN MODEL (src/model.py)")
    print("=" * 70)

    # Initialize model
    model = BaselineCNN(in_channels=12, num_classes=6)
    param_counts = model.count_parameters()
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters:     {param_counts['total_parameters']:,}")
    print(f"Trainable Parameters: {param_counts['trainable_parameters']:,}")
    print("-" * 70)

    # Test 1: Standard input shape (batch_size=8, channels=12, seq_len=128)
    batch_size, seq_len = 8, 128
    dummy_input = torch.randn(batch_size, 12, seq_len)
    print(f"Input Shape (N, C, L): {dummy_input.shape}")

    logits = model(dummy_input)
    print(f"Output Logits Shape:   {logits.shape} (Expected: ({batch_size}, 6))")
    assert logits.shape == (batch_size, 6), f"Shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values!"
    print("  -> Passed: Standard Forward Pass & Output Shape")

    # Test 2: Permuted input shape (batch_size=8, seq_len=128, channels=12)
    dummy_permuted = torch.randn(batch_size, seq_len, 12)
    logits_permuted = model(dummy_permuted)
    assert logits_permuted.shape == (batch_size, 6)
    print("  -> Passed: Automatic (N, L, C) Permutation")

    # Test 3: Inference methods (predict and predict_proba)
    preds = model.predict(dummy_input)
    probs = model.predict_proba(dummy_input)
    assert preds.shape == (batch_size,)
    assert probs.shape == (batch_size, 6)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    print("  -> Passed: Predict & Probability Softmax Normalization")

    # Test 4: Loss Criterion and Optimizer initialization
    criterion = model.get_loss_criterion()
    optimizer = model.get_optimizer(learning_rate=1e-3)
    dummy_targets = torch.randint(0, 6, (batch_size,))
    loss = criterion(logits, dummy_targets)
    assert torch.isfinite(loss).item()
    print(f"  -> Passed: Loss Criterion ({criterion.__class__.__name__}) Loss={loss.item():.4f}")
    print(f"  -> Passed: Optimizer ({optimizer.__class__.__name__})")

    # Test 5: Variable sequence lengths (e.g., 64, 200, 256)
    for length in [64, 200, 256]:
        var_input = torch.randn(4, 12, length)
        var_out = model(var_input)
        assert var_out.shape == (4, 6)
        assert torch.isfinite(var_out).all()
    print("  -> Passed: Adaptive Pooling across Variable Window Lengths (64, 200, 256)")

    print("=" * 70)
    print("  ALL BASELINE CNN TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_baseline_cnn()

