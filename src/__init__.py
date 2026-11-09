"""
Adaptive Sensor Fusion for Human Activity Recognition (HAR)
Phase 3: Model Architecture & Implementation Package.

Exported Classes & Functions:
    - BaselineCNN: Single-stream 1D CNN baseline model
    - AdaptiveSensorFusion: Dual-branch reliability-weighted sensor fusion model
    - SensorBranchEncoder: Modality-specific 1D CNN branch encoder
    - AdaptiveFusionModule: Dynamic reliability and availability weighting module
    - save_model, load_model: Checkpoint utilities for BaselineCNN
    - save_adaptive_model, load_adaptive_model: Checkpoint utilities for AdaptiveSensorFusion
"""

from src.model import (
    ACTIVITY_CLASSES,
    BaselineCNN,
    NUM_ACTIVITIES,
    NUM_SENSOR_FEATURES,
    SENSOR_COLUMNS,
    load_model,
    save_model,
)
from src.sensor_fusion import (
    ACC_CHANNEL_NAMES,
    ACC_INDICES,
    AdaptiveFusionModule,
    AdaptiveSensorFusion,
    GYRO_CHANNEL_NAMES,
    GYRO_INDICES,
    SensorBranchEncoder,
    load_adaptive_model,
    save_adaptive_model,
)

__all__ = [
    "BaselineCNN",
    "AdaptiveSensorFusion",
    "SensorBranchEncoder",
    "AdaptiveFusionModule",
    "save_model",
    "load_model",
    "save_adaptive_model",
    "load_adaptive_model",
    "SENSOR_COLUMNS",
    "ACTIVITY_CLASSES",
    "NUM_SENSOR_FEATURES",
    "NUM_ACTIVITIES",
    "ACC_INDICES",
    "GYRO_INDICES",
    "ACC_CHANNEL_NAMES",
    "GYRO_CHANNEL_NAMES",
]

