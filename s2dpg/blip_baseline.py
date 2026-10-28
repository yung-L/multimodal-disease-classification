from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from s2dpg.common import normalize


class BlipImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as handle:
            image = handle.convert("RGB")
        return image, int(sample.label), sample.path


def make_image_collate(processor):
    def collate(batch):
        images, labels, paths = zip(*batch)
        pixel_values = processor(
            images=list(images), return_tensors="pt"
        )["pixel_values"]
        return pixel_values, torch.tensor(labels), list(paths)
    return collate


@torch.inference_mode()
def encode_images(model, processor, samples, cfg, device):
    loader = DataLoader(
        BlipImageDataset(samples),
        batch_size=int(cfg["image_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        collate_fn=make_image_collate(processor),
    )
    features, labels, paths = [], [], []
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    model.eval()
    for pixels, target, batch_paths in tqdm(loader, desc="BLIP images"):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            vision = model.vision_model(pixel_values=pixels)
            projected = model.vision_proj(
                vision.last_hidden_state[:, 0, :]
            )
        features.append(normalize(projected).cpu())
        labels.append(target.cpu())
        paths.extend(batch_paths)
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels).long(),
        "paths": paths,
    }


def _bundle_index(bundle: dict) -> Dict[str, dict]:
    classes = bundle.get("classes")
    if not isinstance(classes, list):
        raise ValueError("Source bundle must contain a 'classes' list")
    return {
        str(item["class_name"]): item
        for item in classes
        if item.get("class_name")
    }


def descriptions_by_class(
    bundle: dict,
    classnames: Sequence[str],
    max_per_class: int = 0,
) -> List[List[str]]:
    index = _bundle_index(bundle)
    grouped = []
    for classname in classnames:
        if classname not in index:
            raise KeyError("Class missing from source bundle: %s" % classname)
        values = index[classname].get("source_descriptions", [])
        values = [str(value).strip() for value in values if str(value).strip()]
        if not values:
            values = ["a plant disease image of %s" % classname]
        if max_per_class > 0:
            values = values[:max_per_class]
        grouped.append(values)
    return grouped


@torch.inference_mode()
def encode_text_prototypes(
    model,
    processor,
    grouped_descriptions: Sequence[Sequence[str]],
    cfg,
    device,
):
    flat, labels = [], []
    for class_id, descriptions in enumerate(grouped_descriptions):
        flat.extend(descriptions)
        labels.extend([class_id] * len(descriptions))
    encoded = []
    batch_size = int(cfg["text_batch_size"])
    max_length = int(cfg.get("text_max_length", 35))
    model.eval()
    for start in tqdm(range(0, len(flat), batch_size), desc="BLIP texts"):
        tokens = processor(
            text=flat[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
            if key in {"input_ids", "attention_mask"}
        }
        text_output = model.text_encoder(**tokens)
        projected = model.text_proj(
            text_output.last_hidden_state[:, 0, :]
        )
        encoded.append(normalize(projected).cpu())
    description_features = torch.cat(encoded)
    description_labels = torch.tensor(labels, dtype=torch.long)
    prototypes = []
    for class_id in range(len(grouped_descriptions)):
        selected = description_features[description_labels == class_id]
        prototypes.append(normalize(selected.mean(dim=0)))
    return torch.stack(prototypes), {
        "num_descriptions": len(flat),
        "descriptions_per_class": [len(values) for values in grouped_descriptions],
    }


class BlipPrototypeAdapter(nn.Module):
    def __init__(self, prototypes: torch.Tensor, cfg: dict):
        super().__init__()
        dimension = int(prototypes.shape[1])
        self.register_buffer("prototypes", normalize(prototypes))
        self.adapter = nn.Sequential(
            nn.Linear(dimension, int(cfg["hidden_dim"])),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(int(cfg["hidden_dim"]), dimension),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        maximum = float(cfg["max_residual_ratio"])
        initial = float(cfg["init_residual_ratio"])
        probability = min(max(initial / maximum, 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(
            torch.tensor(math.log(probability / (1.0 - probability)))
        )
        self.max_residual_ratio = maximum
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(float(cfg["init_logit_scale"])))
        )

    def forward(self, image_features):
        image_features = normalize(image_features)
        ratio = self.max_residual_ratio * self.residual_logit.sigmoid()
        adapted = normalize(
            (1.0 - ratio) * image_features
            + ratio * self.adapter(image_features)
        )
        return (
            self.logit_scale.exp().clamp(max=100.0)
            * adapted
            @ self.prototypes.t()
        )


@torch.inference_mode()
def predict(model, payload, device, batch_size):
    loader = DataLoader(
        TensorDataset(payload["features"], payload["labels"]),
        batch_size=batch_size,
        shuffle=False,
    )
    logits, labels = [], []
    model.eval()
    for features, target in loader:
        logits.append(model(features.to(device)).cpu())
        labels.append(target.cpu())
    return torch.cat(logits), torch.cat(labels)


def train_adapter(cache, prototypes, cfg, device, metric_fn):
    model = BlipPrototypeAdapter(prototypes, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(torch.initial_seed())
    train_loader = DataLoader(
        TensorDataset(
            cache["train"]["features"], cache["train"]["labels"]
        ),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    best_state, best_score, best_epoch = None, -float("inf"), 0
    patience_left = int(cfg["patience"])
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total_loss, total_items = 0.0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(features), labels,
                label_smoothing=float(cfg["label_smoothing"]),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_items += len(labels)
        val_logits, val_labels = predict(
            model, cache["val"], device, int(cfg["batch_size"])
        )
        val_metrics, _ = metric_fn(val_logits, val_labels)
        score = float(val_metrics["macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": score,
        })
        print(
            "Epoch %03d loss=%.4f val_acc=%.2f val_f1=%.2f"
            % (epoch, history[-1]["train_loss"], val_metrics["accuracy"], score)
        )
        if score > best_score + 1e-8:
            best_score, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = int(cfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("BLIP adapter training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_score
