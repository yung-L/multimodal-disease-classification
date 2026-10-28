from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip

from .common import normalize, stable_hash
from .data import RegionalImageDataset, Sample


def load_clip(backbone: str, device: torch.device):
    model, preprocess = clip.load(backbone, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


def encode_regional_images(
    model,
    preprocess,
    samples: Sequence[Sample],
    region_cfg: dict,
    batch_size: int,
    workers: int,
    device: torch.device,
    amp: bool,
    augment: bool = False,
    input_size: int = 224,
) -> Dict[str, object]:
    loader = DataLoader(
        RegionalImageDataset(
            samples,
            preprocess,
            region_cfg,
            augment=augment,
            input_size=input_size,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features, labels, paths = [], [], []
    with torch.inference_mode():
        for images, target, image_paths in tqdm(loader, desc="Encoding regional images"):
            batch, regions, channels, height, width = images.shape
            images = images.reshape(batch * regions, channels, height, width).to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                encoded = model.encode_image(images)
            encoded = normalize(encoded).reshape(batch, regions, -1).cpu()
            features.append(encoded)
            labels.append(target.long())
            paths.extend(image_paths)
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "paths": paths,
    }


def encode_symptom_prompts(model, graph: dict, device: torch.device) -> torch.Tensor:
    encoded: List[torch.Tensor] = []
    with torch.inference_mode():
        for symptom in tqdm(graph["symptoms"], desc="Encoding symptom prompts"):
            prompts = symptom["prompts"]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            features = normalize(model.encode_text(tokens))
            encoded.append(normalize(features.mean(dim=0)).cpu())
    return torch.stack(encoded)


def encode_class_descriptions(
    model,
    source_bundle: dict,
    classnames: Sequence[str],
    device: torch.device,
):
    source_map = {
        item["class_name"]: item["source_descriptions"]
        for item in source_bundle["classes"]
    }
    features, labels, texts = [], [], []
    with torch.inference_mode():
        for class_id, classname in enumerate(
            tqdm(classnames, desc="Encoding official class descriptions")
        ):
            descriptions = source_map.get(classname, [])
            if not descriptions:
                descriptions = [
                    "A photo of %s, a type of plant disease." % classname
                ]
            tokens = clip.tokenize(descriptions, truncate=True).to(device)
            class_features = normalize(model.encode_text(tokens)).cpu()
            features.append(class_features)
            labels.append(
                torch.full(
                    (len(class_features),), class_id, dtype=torch.long
                )
            )
            texts.append(descriptions)
    return torch.cat(features), torch.cat(labels), texts


def prepare_cache(
    cfg: dict,
    split: Dict[str, Sequence[Sample]],
    classnames: Sequence[str],
    graph: dict,
    device: torch.device,
    force: bool = False,
) -> Path:
    cache_root = Path(cfg["project"]["cache_dir"])
    cache_root.mkdir(parents=True, exist_ok=True)
    region_cfg = cfg["clip"]["regions"]
    source_bundle = json.load(
        open(cfg["data"]["source_bundle"], encoding="utf-8")
    )
    source_hash = stable_hash(
        json.dumps(source_bundle, ensure_ascii=False, sort_keys=True)
    )
    augment_views = int(cfg["clip"].get("train_augment_views", 1))
    cache_key = stable_hash(
        str(Path(cfg["data"]["root"]).expanduser().resolve()),
        cfg["clip"]["backbone"],
        str(cfg["data"]["split_seed"]),
        str(region_cfg),
        str(augment_views),
        source_hash,
        graph["graph_hash"],
    )
    path = cache_root / ("plantwild_s2dpg_%s.pt" % cache_key)
    if path.exists() and not force:
        print("Using feature cache: %s" % path)
        return path
    model, preprocess = load_clip(cfg["clip"]["backbone"], device)
    input_size = int(getattr(model.visual, "input_resolution", 224))
    payload = {
        "classnames": list(classnames),
        "graph_hash": graph["graph_hash"],
        "backbone": cfg["clip"]["backbone"],
    }
    train_payloads = []
    for view_id in range(augment_views):
        print(
            "Encoding augmented training view %d/%d"
            % (view_id + 1, augment_views)
        )
        train_payloads.append(encode_regional_images(
            model,
            preprocess,
            split["train"],
            region_cfg,
            int(cfg["clip"]["image_batch_size"]),
            int(cfg["clip"]["num_workers"]),
            device,
            bool(cfg["clip"].get("amp", True)),
            augment=True,
            input_size=input_size,
        ))
    payload["train"] = {
        "features": torch.cat(
            [item["features"] for item in train_payloads]
        ),
        "labels": torch.cat([item["labels"] for item in train_payloads]),
        "paths": sum([item["paths"] for item in train_payloads], []),
    }
    for split_name in ("val", "test"):
        payload[split_name] = encode_regional_images(
            model,
            preprocess,
            split[split_name],
            region_cfg,
            int(cfg["clip"]["image_batch_size"]),
            int(cfg["clip"]["num_workers"]),
            device,
            bool(cfg["clip"].get("amp", True)),
            augment=False,
            input_size=input_size,
        )
    payload["symptom_features"] = encode_symptom_prompts(model, graph, device)
    (
        payload["class_text_features"],
        payload["class_text_labels"],
        payload["class_texts"],
    ) = encode_class_descriptions(
        model, source_bundle, classnames, device
    )
    torch.save(payload, path)
    print("Saved feature cache: %s" % path)
    return path
