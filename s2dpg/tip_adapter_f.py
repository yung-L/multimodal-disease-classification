from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from s2dpg.common import normalize


class TipAdapterF(nn.Module):
    def __init__(self, cache_keys, cache_values, text_prototypes, logit_scale):
        super().__init__()
        feature_dim, cache_size = cache_keys.shape
        self.adapter = nn.Linear(feature_dim, cache_size, bias=False)
        with torch.no_grad():
            self.adapter.weight.copy_(cache_keys.t())
        self.register_buffer("cache_values", cache_values.float())
        self.register_buffer("text_prototypes", normalize(text_prototypes))
        self.register_buffer(
            "fixed_logit_scale", torch.tensor(float(logit_scale)).clamp(max=100.0)
        )

    def components(self, image_features, beta):
        features = normalize(image_features)
        affinity = self.adapter(features)
        cache_logits = torch.exp(-float(beta) * (1.0 - affinity)) @ self.cache_values
        clip_logits = self.fixed_logit_scale * features @ self.text_prototypes.t()
        return clip_logits, cache_logits

    def forward(self, image_features, beta, alpha):
        clip_logits, cache_logits = self.components(image_features, beta)
        return clip_logits + float(alpha) * cache_logits


@torch.inference_mode()
def predict(model, payload, device, batch_size, beta, alpha):
    loader = DataLoader(
        TensorDataset(payload["features"], payload["labels"]),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    logits, labels = [], []
    for features, target in loader:
        logits.append(model(features.to(device), beta, alpha).cpu())
        labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def train_tip_adapter_f(model, cache, cfg, device, metric_fn, seed):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(cfg["learning_rate"]),
        eps=1e-4,
        weight_decay=float(cfg["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(cache["train"]["features"], cache["train"]["labels"]),
        batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(cfg["train_epochs"]) * len(loader))
    )
    beta, alpha = float(cfg["init_beta"]), float(cfg["init_alpha"])
    best_state, best_score, best_epoch = None, -float("inf"), 0
    stale, history = 0, []
    for epoch in range(1, int(cfg["train_epochs"]) + 1):
        model.train()
        total, count = 0.0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, beta, alpha)
            loss = F.cross_entropy(
                logits, labels,
                label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(labels)
            count += len(labels)
        val_logits, val_labels = predict(
            model, cache["val"], device, int(cfg["batch_size"]), beta, alpha
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
        raise RuntimeError("Tip-Adapter-F produced no checkpoint")
    model.adapter.load_state_dict(best_state)
    return model, history, best_epoch, best_score


@torch.inference_mode()
def search_hyperparameters(model, val_payload, cfg, device, metric_fn):
    features = normalize(val_payload["features"].to(device))
    labels = val_payload["labels"]
    affinity = model.adapter(features)
    clip_logits = (
        model.fixed_logit_scale * features @ model.text_prototypes.t()
    ).cpu()
    affinity = affinity.cpu()
    values = model.cache_values.cpu()
    betas = torch.linspace(
        float(cfg["beta_min"]), float(cfg["beta_max"]),
        int(cfg["beta_steps"]),
    ).tolist()
    alphas = torch.linspace(
        float(cfg["alpha_min"]), float(cfg["alpha_max"]),
        int(cfg["alpha_steps"]),
    ).tolist()
    best = {"macro_f1": -float("inf"), "accuracy": -float("inf")}
    for beta in betas:
        cache_logits = torch.exp(-beta * (1.0 - affinity)) @ values
        for alpha in alphas:
            metrics, _ = metric_fn(clip_logits + alpha * cache_logits, labels)
            candidate = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            current = (best["macro_f1"], best["accuracy"])
            if candidate > current:
                best = {
                    "beta": float(beta), "alpha": float(alpha),
                    "macro_f1": candidate[0], "accuracy": candidate[1],
                }
    return best
