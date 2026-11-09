#!.venv/Scripts/python.exe
"""
src/generate_diagram.py

Generates the architecture visualization diagram for the MotionSense Human Activity
Recognition (HAR) project:
    1. Baseline 1D CNN Architecture (Single-stream 12-channel)
    2. Adaptive Sensor Fusion Architecture (Dual-branch reliability-weighted)

Saves the output to diagrams/model_architecture.png.
"""

import os
import tempfile
from pathlib import Path

# Set writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_har_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Output directory setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DIAGRAMS_DIR / "model_architecture.png"


def draw_architecture_diagram():
    # Set high-DPI figure
    fig = plt.figure(figsize=(18, 12), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title Banner
    ax.text(
        50, 97.5,
        "Human Activity Recognition (MotionSense): Model Architectures",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1A202C"
    )
    ax.text(
        50, 95.0,
        "Phase 3: Baseline 1D CNN vs. Dual-Branch Adaptive Sensor Fusion with Dynamic Reliability Weighting",
        ha="center", va="center", fontsize=12, color="#4A5568", style="italic"
    )

    # -------------------------------------------------------------------------
    # LEFT PANEL: Baseline 1D CNN
    # -------------------------------------------------------------------------
    panel_left = patches.FancyBboxPatch(
        (2, 4), 38, 88, boxstyle="round,pad=1",
        facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.5
    )
    ax.add_patch(panel_left)

    ax.text(
        21, 90, "BASELINE 1D CNN MODEL",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#2D3748"
    )
    ax.text(
        21, 87.5, "Unified Input Processing (Single Stream)",
        ha="center", va="center", fontsize=10, color="#718096"
    )

    # Input Box
    box_b_in = patches.FancyBboxPatch(
        (6, 76), 30, 8, boxstyle="round,pad=0.5",
        facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.5
    )
    ax.add_patch(box_b_in)
    ax.text(21, 81.5, "Input: MotionSense Sensor Window", ha="center", va="center", fontsize=10, fontweight="bold", color="#2B6CB0")
    ax.text(21, 78.5, "Shape: (Batch, 12 Channels, Length L)\n12 Scaled Motion Features", ha="center", va="center", fontsize=8.5, color="#2D3748")

    # Arrow 1
    ax.annotate("", xy=(21, 70), xytext=(21, 76), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Conv Block 1
    box_b_c1 = patches.FancyBboxPatch(
        (6, 62), 30, 8, boxstyle="round,pad=0.5",
        facecolor="#EDF2F7", edgecolor="#4A5568", linewidth=1.2
    )
    ax.add_patch(box_b_c1)
    ax.text(21, 67, "Conv1D Block 1", ha="center", va="center", fontsize=10, fontweight="bold", color="#2D3748")
    ax.text(21, 64, "Conv1D(12 -> 64, k=5, p=2) + BatchNorm1D\nReLU + MaxPool1D(2) + Dropout(0.2)", ha="center", va="center", fontsize=8, color="#4A5568")

    # Arrow 2
    ax.annotate("", xy=(21, 56), xytext=(21, 62), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Conv Block 2
    box_b_c2 = patches.FancyBboxPatch(
        (6, 48), 30, 8, boxstyle="round,pad=0.5",
        facecolor="#EDF2F7", edgecolor="#4A5568", linewidth=1.2
    )
    ax.add_patch(box_b_c2)
    ax.text(21, 53, "Conv1D Block 2", ha="center", va="center", fontsize=10, fontweight="bold", color="#2D3748")
    ax.text(21, 50, "Conv1D(64 -> 128, k=5, p=2) + BatchNorm1D\nReLU + MaxPool1D(2) + Dropout(0.2)", ha="center", va="center", fontsize=8, color="#4A5568")

    # Arrow 3
    ax.annotate("", xy=(21, 42), xytext=(21, 48), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Conv Block 3
    box_b_c3 = patches.FancyBboxPatch(
        (6, 34), 30, 8, boxstyle="round,pad=0.5",
        facecolor="#EDF2F7", edgecolor="#4A5568", linewidth=1.2
    )
    ax.add_patch(box_b_c3)
    ax.text(21, 39, "Conv1D Block 3 + Pooling", ha="center", va="center", fontsize=10, fontweight="bold", color="#2D3748")
    ax.text(21, 36, "Conv1D(128 -> 256, k=3, p=1) + BatchNorm1D\nReLU + AdaptiveAvgPool1D(1) + Dropout(0.3)", ha="center", va="center", fontsize=8, color="#4A5568")

    # Arrow 4
    ax.annotate("", xy=(21, 28), xytext=(21, 34), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Dense Classifier
    box_b_cls = patches.FancyBboxPatch(
        (6, 20), 30, 8, boxstyle="round,pad=0.5",
        facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=1.5
    )
    ax.add_patch(box_b_cls)
    ax.text(21, 25, "Fully Connected Classifier", ha="center", va="center", fontsize=10, fontweight="bold", color="#744210")
    ax.text(21, 22, "Linear(256 -> 128) + ReLU + Dropout(0.3)\nLinear(128 -> 6 Activity Classes)", ha="center", va="center", fontsize=8, color="#744210")

    # Arrow 5
    ax.annotate("", xy=(21, 14), xytext=(21, 20), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Output Box
    box_b_out = patches.FancyBboxPatch(
        (8, 7), 26, 7, boxstyle="round,pad=0.5",
        facecolor="#C6F6D5", edgecolor="#38A169", linewidth=1.5
    )
    ax.add_patch(box_b_out)
    ax.text(21, 11.5, "Activity Logits (Batch, 6)", ha="center", va="center", fontsize=10, fontweight="bold", color="#22543D")
    ax.text(21, 9, "Loss: CrossEntropyLoss | Opt: Adam\nParams: 177,670", ha="center", va="center", fontsize=8, color="#22543D")

    # -------------------------------------------------------------------------
    # RIGHT PANEL: Adaptive Sensor Fusion Model
    # -------------------------------------------------------------------------
    panel_right = patches.FancyBboxPatch(
        (43, 4), 55, 88, boxstyle="round,pad=1",
        facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.5
    )
    ax.add_patch(panel_right)

    ax.text(
        70.5, 90, "ADAPTIVE SENSOR FUSION MODEL",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#2D3748"
    )
    ax.text(
        70.5, 87.5, "Dual-Branch Encoder with Dynamic Reliability & Availability Gating",
        ha="center", va="center", fontsize=10, color="#718096"
    )

    # Branch 1 Input Box (Accelerometer + Gravity)
    box_f_in1 = patches.FancyBboxPatch(
        (46, 76), 22, 8, boxstyle="round,pad=0.5",
        facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.5
    )
    ax.add_patch(box_f_in1)
    ax.text(57, 81.5, "Branch 1: Accel & Gravity", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2B6CB0")
    ax.text(57, 78.5, "Channels: 6 (gravity + userAcc)\nShape: (Batch, 6, L)", ha="center", va="center", fontsize=8, color="#2D3748")

    # Branch 2 Input Box (Gyroscope + Attitude)
    box_f_in2 = patches.FancyBboxPatch(
        (73, 76), 22, 8, boxstyle="round,pad=0.5",
        facecolor="#FAF5FF", edgecolor="#805AD5", linewidth=1.5
    )
    ax.add_patch(box_f_in2)
    ax.text(84, 81.5, "Branch 2: Gyro & Attitude", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#6B46C1")
    ax.text(84, 78.5, "Channels: 6 (attitude + rotRate)\nShape: (Batch, 6, L)", ha="center", va="center", fontsize=8, color="#2D3748")

    # Arrows to Encoders
    ax.annotate("", xy=(57, 70), xytext=(57, 76), arrowprops=dict(arrowstyle="->", color="#3182CE", lw=1.5))
    ax.annotate("", xy=(84, 70), xytext=(84, 76), arrowprops=dict(arrowstyle="->", color="#805AD5", lw=1.5))

    # Branch 1 Encoder Box
    box_f_enc1 = patches.FancyBboxPatch(
        (46, 57), 22, 13, boxstyle="round,pad=0.5",
        facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.2
    )
    ax.add_patch(box_f_enc1)
    ax.text(57, 67.5, "SensorBranchEncoder 1", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2B6CB0")
    ax.text(57, 63, "Conv1D(6 -> 64, k=5) + BN + ReLU\nConv1D(64 -> 128, k=5) + BN + ReLU\nConv1D(128 -> 128, k=3) + Pool\nProjector -> Embedding h_acc (128)", ha="center", va="center", fontsize=7.5, color="#2D3748")

    # Branch 2 Encoder Box
    box_f_enc2 = patches.FancyBboxPatch(
        (73, 57), 22, 13, boxstyle="round,pad=0.5",
        facecolor="#FAF5FF", edgecolor="#805AD5", linewidth=1.2
    )
    ax.add_patch(box_f_enc2)
    ax.text(84, 67.5, "SensorBranchEncoder 2", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#6B46C1")
    ax.text(84, 63, "Conv1D(6 -> 64, k=5) + BN + ReLU\nConv1D(64 -> 128, k=5) + BN + ReLU\nConv1D(128 -> 128, k=3) + Pool\nProjector -> Embedding h_gyro (128)", ha="center", va="center", fontsize=7.5, color="#2D3748")

    # Availability & Reliability Indicators Box
    box_diag = patches.FancyBboxPatch(
        (48, 48), 45, 6, boxstyle="round,pad=0.3",
        facecolor="#FFF5F5", edgecolor="#E53E3E", linewidth=1.2, linestyle="--"
    )
    ax.add_patch(box_diag)
    ax.text(70.5, 51.7, "Sensor Reliability & Availability Monitoring Indicators", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#C53030")
    ax.text(70.5, 49.3, "Masks: m_acc, m_gyro in {0, 1} | Reliability: r_acc, r_gyro in [0, 1] | Div-by-Zero Guard (eps=1e-8)", ha="center", va="center", fontsize=7.5, color="#742A2A")

    # Arrows to Adaptive Fusion Module
    ax.annotate("", xy=(62, 42), xytext=(57, 57), arrowprops=dict(arrowstyle="->", color="#3182CE", lw=1.5))
    ax.annotate("", xy=(79, 42), xytext=(84, 57), arrowprops=dict(arrowstyle="->", color="#805AD5", lw=1.5))
    ax.annotate("", xy=(70.5, 42), xytext=(70.5, 48), arrowprops=dict(arrowstyle="->", color="#E53E3E", lw=1.2, linestyle="--"))

    # Adaptive Fusion Module Box
    box_fusion = patches.FancyBboxPatch(
        (46, 30), 49, 12, boxstyle="round,pad=0.5",
        facecolor="#FED7D7", edgecolor="#E53E3E", linewidth=1.5
    )
    ax.add_patch(box_fusion)
    ax.text(70.5, 39.5, "ADAPTIVE FUSION MODULE (Reliability-Weighted Gating)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#9B2C2C")
    ax.text(70.5, 36.2, r"Effective Reliability: R_i = m_i * r_i   |   Content Importance: s_i = G(h_i)", ha="center", va="center", fontsize=8.5, color="#2D3748")
    ax.text(70.5, 33.2, r"Adaptive Weights: w_i = (R_i * exp(s_i)) / (sum R_k * exp(s_k) + eps)", ha="center", va="center", fontsize=8.2, color="#9B2C2C")
    ax.text(70.5, 31.0, "Boundary Rules: m_acc=0 -> w_gyro=1.0, w_acc=0.0  |  m_gyro=0 -> w_acc=1.0, w_gyro=0.0", ha="center", va="center", fontsize=7.8, color="#2D3748")

    # Arrow from Fusion to Classifier
    ax.annotate("", xy=(70.5, 24), xytext=(70.5, 30), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Fusion Classifier Box
    box_f_cls = patches.FancyBboxPatch(
        (52, 17), 37, 7, boxstyle="round,pad=0.5",
        facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=1.5
    )
    ax.add_patch(box_f_cls)
    ax.text(70.5, 21.5, "Classification Head (Fused Stream)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#744210")
    ax.text(70.5, 18.8, "Linear(256 -> 128) + ReLU + Dropout(0.3) + Linear(128 -> 6)", ha="center", va="center", fontsize=8, color="#744210")

    # Arrow to Output
    ax.annotate("", xy=(70.5, 13), xytext=(70.5, 17), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))

    # Fusion Output Box
    box_f_out = patches.FancyBboxPatch(
        (55, 6), 31, 7, boxstyle="round,pad=0.5",
        facecolor="#C6F6D5", edgecolor="#38A169", linewidth=1.5
    )
    ax.add_patch(box_f_out)
    ax.text(70.5, 10.5, "Activity Logits: Shape (Batch, 6)", ha="center", va="center", fontsize=10, fontweight="bold", color="#22543D")
    ax.text(70.5, 8.0, "Loss: CrossEntropyLoss | Optimizer: Adam\nParams: 265,032 (Branch Encoders: 218K | Fusion: 12.5K)", ha="center", va="center", fontsize=7.5, color="#22543D")

    # Save diagram
    plt.savefig(OUTPUT_FILE, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram successfully generated and saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    draw_architecture_diagram()
