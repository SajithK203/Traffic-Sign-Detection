"""
visualization.py
-----------------
Utilities for plotting results: PR curves, confusion matrices, qualitative grids.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


# ------------------------------------------------------------------
# Precision-Recall curve
# ------------------------------------------------------------------

def plot_pr_curve(
    precisions_dict: dict[str, np.ndarray],
    recalls_dict:    dict[str, np.ndarray],
    ap_dict:         dict[str, float],
    out_path:        str | Path = "results/figures/pr_curve.png",
    title:           str = "Precision–Recall Curve",
) -> None:
    """Plot precision-recall curves for multiple models on the same axes.

    Args:
        precisions_dict: {model_name: precision_array}
        recalls_dict:    {model_name: recall_array}
        ap_dict:         {model_name: AP value}
        out_path:        Where to save the figure.
        title:           Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in precisions_dict:
        label = f"{name} (AP={ap_dict.get(name, 0):.3f})"
        ax.plot(recalls_dict[name], precisions_dict[name], label=label, lw=2)

    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(title,        fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[VIZ] PR curve saved → {out_path}")


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:    list | np.ndarray,
    y_pred:    list | np.ndarray,
    class_names: list[str],
    out_path:  str | Path = "results/figures/confusion_matrix.png",
    title:     str = "Confusion Matrix",
) -> None:
    """Plot and save a normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, normalize="true",
        xticks_rotation="vertical", ax=ax, colorbar=False,
    )
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[VIZ] Confusion matrix saved → {out_path}")


# ------------------------------------------------------------------
# Qualitative image grid
# ------------------------------------------------------------------

def save_qualitative_grid(
    images:   list[np.ndarray],
    titles:   list[str],
    out_path: str | Path = "results/qualitative_examples/grid.png",
    n_cols:   int = 4,
    img_size: tuple[int, int] = (320, 240),
) -> None:
    """Save a grid of annotated images as a single figure.

    Args:
        images:   List of BGR images (numpy arrays).
        titles:   Per-image caption strings.
        out_path: Output path for the grid PNG.
        n_cols:   Number of columns in the grid.
        img_size: (width, height) to resize each thumbnail.
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            rgb = cv2.cvtColor(
                cv2.resize(images[i], img_size), cv2.COLOR_BGR2RGB
            )
            ax.imshow(rgb)
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[VIZ] Qualitative grid saved → {out_path}")


# ------------------------------------------------------------------
# Class distribution bar chart
# ------------------------------------------------------------------

def plot_class_distribution(
    class_counts: dict[str, int],
    out_path: str | Path = "results/figures/class_distribution.png",
    title: str = "Class Distribution",
) -> None:
    """Bar chart of per-class instance counts."""
    names  = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    bars = ax.bar(range(len(names)), counts, color=sns.color_palette("muted", len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Instance Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[VIZ] Class distribution chart saved → {out_path}")
