from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int) -> float:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    value = torch.zeros((), device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            value += mask.float().mean() * (
                confidence[mask].mean() - correct[mask].float().mean()
            ).abs()
    return float(value)


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ece_bins: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    logits, labels = logits.float(), labels.long()
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    y_true, y_pred = labels.cpu().numpy(), predictions.cpu().numpy()
    classes = np.arange(num_classes)
    topk = min(5, num_classes)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "macro_f1": 100.0 * f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "macro_precision": 100.0 * precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "macro_recall": 100.0 * recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "top5_accuracy": 100.0 * float(logits.topk(topk, dim=1).indices.eq(labels[:, None]).any(dim=1).float().mean()),
        "ece": expected_calibration_error(probabilities, labels, ece_bins),
        "nll": float(F.cross_entropy(logits, labels)),
        "brier": float(((probabilities - one_hot) ** 2).sum(dim=1).mean()),
    }, confusion_matrix(y_true, y_pred, labels=classes)


def long_tail_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    counts: Iterable[int],
    quantiles,
) -> Dict[str, dict]:
    counts_array = np.asarray(list(counts))
    order = np.argsort(counts_array)
    first = max(1, int(round(len(order) * quantiles[0])))
    second = max(first + 1, int(round(len(order) * quantiles[1])))
    groups = {"tail": order[:first], "medium": order[first:second], "head": order[second:]}
    prediction = logits.argmax(dim=1)
    output = {}
    for name, class_ids in groups.items():
        class_tensor = torch.as_tensor(class_ids, device=labels.device)
        mask = (labels[:, None] == class_tensor[None, :]).any(dim=1)
        true = labels[mask].cpu().numpy()
        pred = prediction[mask].cpu().numpy()
        output[name] = {
            "accuracy": 100.0 * accuracy_score(true, pred),
            "macro_f1": 100.0 * f1_score(true, pred, labels=class_ids, average="macro", zero_division=0),
            "classes": class_ids.tolist(),
        }
    return output
