from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_confusion(
    matrix: np.ndarray,
    path: str,
    title: str = "S2-DPG Confusion Matrix",
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_reliability(logits: torch.Tensor, labels: torch.Tensor, path: str, bins: int) -> None:
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    edges = torch.linspace(0, 1, bins + 1)
    accuracy, average_confidence, centers = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = confidence.gt(lower) & confidence.le(upper)
        centers.append(float((lower + upper) / 2))
        accuracy.append(float(correct[mask].float().mean()) if mask.any() else 0)
        average_confidence.append(float(confidence[mask].mean()) if mask.any() else 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(centers, accuracy, marker="o", label="Accuracy")
    ax.plot(centers, average_confidence, marker="s", label="Confidence")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
