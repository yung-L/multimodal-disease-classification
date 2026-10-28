from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .common import normalize, stable_hash
from .data import Sample
from .metrics import classification_metrics


class DinoImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform,
    ):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.label, sample.path


def build_dino_transform(processor, augment: bool):
    size = getattr(processor, "crop_size", None) or processor.size
    if isinstance(size, dict):
        crop_size = int(
            size.get("height", size.get("shortest_edge", 224))
        )
    else:
        crop_size = int(size)
    mean = processor.image_mean
    std = processor.image_std
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.60, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    resize_size = int(round(crop_size / 0.875))
    return transforms.Compose([
        transforms.Resize(
            resize_size, interpolation=InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_dinov2(model_name_or_path: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(
        model_name_or_path
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


@torch.inference_mode()
def encode_dino_split(
    model,
    samples,
    transform,
    batch_size,
    workers,
    device,
    description,
):
    loader = DataLoader(
        DinoImageDataset(samples, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    features, labels, paths = [], [], []
    for images, target, image_paths in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16,
        ):
            output = model(pixel_values=images)
            encoded = output.last_hidden_state[:, 0]
        features.append(normalize(encoded).cpu())
        labels.append(target.long())
        paths.extend(image_paths)
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "paths": paths,
    }


def prepare_dino_cache(
    cfg: dict,
    split: Dict[str, Sequence[Sample]],
    device: torch.device,
    force: bool = False,
):
    section = cfg["dino"]
    cache_dir = Path(section["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = stable_hash(
        str(Path(cfg["data"]["root"]).expanduser().resolve()),
        str(section["model"]),
        str(cfg["data"]["split_seed"]),
        str(section["train_augment_views"]),
    )
    cache_path = cache_dir / ("dinov2_probe_%s.pt" % key)
    if cache_path.exists() and not force:
        print("Using DINOv2 feature cache:", cache_path)
        return torch.load(cache_path, map_location="cpu")

    model, processor = load_dinov2(section["model"], device)
    eval_transform = build_dino_transform(processor, augment=False)
    train_transform = build_dino_transform(processor, augment=True)
    train_views = []
    for view in range(int(section["train_augment_views"])):
        train_views.append(encode_dino_split(
            model,
            split["train"],
            train_transform,
            int(section["image_batch_size"]),
            int(section["num_workers"]),
            device,
            "Encoding DINO train view %d" % (view + 1),
        ))
    payload = {
        "train": {
            "features": torch.cat([
                item["features"] for item in train_views
            ]),
            "labels": torch.cat([
                item["labels"] for item in train_views
            ]),
            "paths": sum([
                item["paths"] for item in train_views
            ], []),
        }
    }
    for name in ("val", "test"):
        payload[name] = encode_dino_split(
            model,
            split[name],
            eval_transform,
            int(section["image_batch_size"]),
            int(section["num_workers"]),
            device,
            "Encoding DINO %s" % name,
        )
    payload["model"] = str(section["model"])
    torch.save(payload, cache_path)
    print("Saved DINOv2 feature cache:", cache_path)
    return payload


class CosineProbe(nn.Module):
    def __init__(self, prototypes, init_scale):
        super().__init__()
        self.weight = nn.Parameter(normalize(prototypes))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(float(init_scale)))
        )

    def forward(self, features):
        return (
            self.logit_scale.exp().clamp(1, 100)
            * normalize(features)
            @ normalize(self.weight).t()
        )


def class_mean_initialization(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
):
    values = []
    global_mean = normalize(features.mean(dim=0))
    for class_id in range(num_classes):
        selected = features[labels == class_id]
        values.append(
            normalize(selected.mean(dim=0))
            if len(selected)
            else global_mean
        )
    return torch.stack(values)


def evaluate_probe(
    model,
    features,
    labels,
    num_classes,
    ece_bins,
    batch_size=2048,
):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            logits.append(model(
                features[start:start + batch_size].to(
                    next(model.parameters()).device
                )
            ).cpu())
    logits = torch.cat(logits)
    metrics, matrix = classification_metrics(
        logits, labels, num_classes, ece_bins
    )
    return logits, metrics, matrix


def train_dino_probe(
    cache,
    num_classes,
    cfg,
    device,
):
    section = cfg["probe"]
    train_features = cache["train"]["features"].float()
    train_labels = cache["train"]["labels"].long()
    val_features = cache["val"]["features"].float()
    val_labels = cache["val"]["labels"].long()
    prototypes = class_mean_initialization(
        train_features, train_labels, num_classes
    )
    model = CosineProbe(
        prototypes, section["init_logit_scale"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["lr"]),
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(section["epochs"])
    )
    counts = torch.bincount(
        train_labels, minlength=num_classes
    ).float().to(device)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=int(section["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    best_state = None
    best_score = -1.0
    best_epoch = -1
    history = []
    for epoch in range(int(section["epochs"])):
        model.train()
        total_loss, seen = 0.0, 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            adjusted = (
                logits
                + float(section["balance_strength"])
                * counts.clamp_min(1).log()[None]
            )
            loss = F.cross_entropy(
                adjusted,
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()
        _, metrics, _ = evaluate_probe(
            model,
            val_features,
            val_labels,
            num_classes,
            int(cfg["evaluation"]["ece_bins"]),
        )
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / seen,
            **metrics,
        })
        print(
            "DINO probe epoch %03d loss=%.4f "
            "val_acc=%.2f val_f1=%.2f"
            % (
                epoch + 1,
                total_loss / seen,
                metrics["accuracy"],
                metrics["macro_f1"],
            )
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "history": history,
    }


def temperature_scale(
    logits: torch.Tensor,
    labels: torch.Tensor,
    minimum: float,
    maximum: float,
    steps: int,
):
    best_temperature, best_nll = 1.0, float("inf")
    for value in torch.linspace(minimum, maximum, steps):
        temperature = float(value)
        nll = float(F.cross_entropy(logits / temperature, labels))
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return best_temperature, best_nll


def search_fusion(
    clip_logits,
    dino_logits,
    labels,
    num_classes,
    ece_bins,
    clip_temperature,
    dino_temperature,
    alpha_steps,
):
    best = None
    for alpha in torch.linspace(0, 1, alpha_steps):
        alpha = float(alpha)
        fused = (
            (1.0 - alpha) * clip_logits / clip_temperature
            + alpha * dino_logits / dino_temperature
        )
        metrics, _ = classification_metrics(
            fused, labels, num_classes, ece_bins
        )
        key = (
            metrics["macro_f1"],
            metrics["accuracy"],
            -alpha,
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "alpha": alpha,
                "metrics": metrics,
            }
    best.pop("key")
    return best
