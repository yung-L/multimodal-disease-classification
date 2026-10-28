from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from s2dpg.common import normalize


class Adapter(nn.Module):
    """Official CLIP-Adapter two-layer bottleneck architecture."""

    def __init__(self, feature_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, feature_dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        return self.fc(features)


class CLIPAdapterClassifier(nn.Module):
    def __init__(self, text_prototypes, logit_scale, cfg):
        super().__init__()
        feature_dim = int(text_prototypes.shape[1])
        self.adapter = Adapter(feature_dim, int(cfg["reduction"]))
        self.register_buffer("text_prototypes", normalize(text_prototypes))
        self.register_buffer(
            "fixed_logit_scale", torch.tensor(float(logit_scale)).clamp(max=100.0)
        )
        self.residual_ratio = float(cfg["residual_ratio"])

    def forward(self, image_features):
        original = normalize(image_features)
        adapted = self.adapter(original)
        mixed = (
            self.residual_ratio * adapted
            + (1.0 - self.residual_ratio) * original
        )
        mixed = normalize(mixed)
        return self.fixed_logit_scale * mixed @ self.text_prototypes.t()


@torch.inference_mode()
def predict(model, payload, device, batch_size):
    loader = DataLoader(
        TensorDataset(payload["features"], payload["labels"]),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    logits, labels = [], []
    for features, target in loader:
        logits.append(model(features.to(device)).cpu())
        labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def train_adapter(model, cache, cfg, device, metric_fn, seed):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(cache["train"]["features"], cache["train"]["labels"]),
        batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(cfg["epochs"]) * len(loader))
    )
    best_state, best_score, best_epoch = None, -float("inf"), 0
    stale, history = 0, []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total, count = 0.0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(
                logits, labels,
                label_smoothing=float(cfg["label_smoothing"]),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(labels)
            count += len(labels)
        val_logits, val_labels = predict(
            model, cache["val"], device, int(cfg["batch_size"])
        )
        val_metrics, _ = metric_fn(val_logits, val_labels)
        score = float(val_metrics["macro_f1"])
        row = {
            "epoch": epoch, "train_loss": total / max(count, 1),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": score,
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f val_acc=%.2f val_f1=%.2f"
            % (epoch, row["train_loss"], row["val_accuracy"], score)
        )
        if score > best_score + 1e-8:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.adapter.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("CLIP-Adapter produced no checkpoint")
    model.adapter.load_state_dict(best_state)
    return model, history, best_epoch, best_score
