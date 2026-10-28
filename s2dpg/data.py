from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    classname: str


def locate_image_root(root: str, image_subdir: str = "images") -> Path:
    root_path = Path(root).expanduser().resolve()
    candidates = [
        root_path / image_subdir,
        root_path / "plantwild" / image_subdir,
        root_path,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(path.is_dir() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError("Cannot locate PlantWild class folders under %s" % root_path)


def discover_dataset(
    root: str,
    image_subdir: str,
    extensions: Iterable[str],
) -> Tuple[List[Sample], List[str]]:
    image_root = locate_image_root(root, image_subdir)
    allowed = {suffix.lower() for suffix in extensions}
    class_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    classnames = [path.name for path in class_dirs]
    samples: List[Sample] = []
    for label, class_dir in enumerate(class_dirs):
        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )
        if not images:
            raise RuntimeError("No images found in %s" % class_dir)
        samples.extend(Sample(str(path), label, class_dir.name) for path in images)
    return samples, classnames


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to one")
    grouped: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)
        if len(items) < 3:
            raise RuntimeError("Class %s has fewer than three images" % items[0].classname)
        n_test = max(1, round(len(items) * test_ratio))
        n_val = max(1, round(len(items) * val_ratio))
        if n_test + n_val >= len(items):
            n_test, n_val = 1, 1
        n_train = len(items) - n_test - n_val
        result["train"].extend(items[:n_train])
        result["val"].extend(items[n_train:n_train + n_val])
        result["test"].extend(items[n_train + n_val:])
    for key in result:
        rng.shuffle(result[key])
    return result


def save_split(split: Dict[str, Sequence[Sample]], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: [asdict(sample) for sample in values] for key, values in split.items()}
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_split(path: str) -> Dict[str, List[Sample]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        key: [Sample(item["path"], int(item["label"]), item["classname"]) for item in values]
        for key, values in payload.items()
    }


def few_shot_indices(labels: torch.Tensor, shots: int, seed: int) -> torch.Tensor:
    if shots <= 0:
        return torch.arange(len(labels))
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for class_id in labels.unique(sorted=True):
        indices = torch.where(labels == class_id)[0]
        order = indices[torch.randperm(len(indices), generator=generator)]
        selected.append(order[: min(shots, len(order))])
    return torch.cat(selected)


class RegionalImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        preprocess,
        region_cfg: dict,
        augment: bool = False,
        input_size: int = 224,
    ):
        self.samples = list(samples)
        self.region_cfg = region_cfg
        if augment:
            self.preprocess = transforms.Compose([
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.55, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])
        else:
            self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _regions(image: Image.Image, cfg: dict) -> List[Image.Image]:
        width, height = image.size
        regions: List[Image.Image] = []
        if cfg.get("global", True):
            regions.append(image)
        if cfg.get("quadrants", True):
            regions.extend([
                image.crop((0, 0, width // 2, height // 2)),
                image.crop((width // 2, 0, width, height // 2)),
                image.crop((0, height // 2, width // 2, height)),
                image.crop((width // 2, height // 2, width, height)),
            ])
        if cfg.get("center", True):
            left, top = width // 4, height // 4
            regions.append(image.crop((left, top, width - left, height - top)))
        return regions

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as handle:
            image = handle.convert("RGB")
            tensors = [self.preprocess(region) for region in self._regions(image, self.region_cfg)]
        return torch.stack(tensors), sample.label, sample.path


class PseudoSymptomRegionalDataset(RegionalImageDataset):
    def __init__(
        self,
        samples,
        preprocess,
        region_cfg,
        annotation_file: str,
        symptom_ids,
        min_judgeability: float,
        min_region_visible_probability: float,
        augment: bool = False,
        input_size: int = 224,
    ):
        super().__init__(
            samples,
            preprocess,
            region_cfg,
            augment=augment,
            input_size=input_size,
        )
        self.symptom_index = {
            value: index for index, value in enumerate(symptom_ids)
        }
        self.region_names = []
        if region_cfg.get("global", True):
            self.region_names.append("global")
        if region_cfg.get("quadrants", True):
            self.region_names.extend([
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ])
        if region_cfg.get("center", True):
            self.region_names.append("center")
        self.region_index = {
            value: index for index, value in enumerate(self.region_names)
        }
        self.min_judgeability = min_judgeability
        self.min_region_visible_probability = (
            min_region_visible_probability
        )
        self.annotations = {}
        with open(annotation_file, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.annotations[str(Path(record["path"]).resolve())] = record

    def __getitem__(self, index: int):
        images, label, path = super().__getitem__(index)
        num_symptoms = len(self.symptom_index)
        target = torch.full((num_symptoms,), -1.0)
        supervision_weight = torch.zeros(num_symptoms)
        region_target = torch.zeros(
            num_symptoms, len(self.region_names)
        )
        region_mask = torch.zeros(num_symptoms, dtype=torch.bool)
        record = self.annotations.get(str(Path(path).resolve()))
        if record is not None:
            for item in record.get("symptoms", []):
                symptom_id = item.get("symptom_id")
                if symptom_id not in self.symptom_index:
                    continue
                symptom_index = self.symptom_index[symptom_id]
                if "visible_probability" in item:
                    visible = max(
                        0.0,
                        float(item.get("visible_probability", 0.0)),
                    )
                    absent = max(
                        0.0,
                        float(item.get("absent_probability", 0.0)),
                    )
                    unjudgeable = max(
                        0.0,
                        float(
                            item.get("unjudgeable_probability", 0.0)
                        ),
                    )
                    total = visible + absent + unjudgeable
                    if total <= 0:
                        continue
                    visible /= total
                    absent /= total
                    unjudgeable /= total
                else:
                    # Backward compatibility for old pilot annotations.
                    value = max(
                        0.0,
                        min(1.0, float(item.get("confidence", 0))),
                    )
                    status = item.get("status")
                    visible = value if status == "visible" else 0.0
                    absent = value if status == "not_visible" else 0.0
                    unjudgeable = max(
                        0.0, 1.0 - visible - absent
                    )

                judgeability = visible + absent
                if judgeability >= self.min_judgeability:
                    target[symptom_index] = visible / max(
                        judgeability, 1e-6
                    )
                    # Suppress weakly judgeable teacher outputs while keeping
                    # genuinely soft visible-vs-absent supervision.
                    supervision_weight[symptom_index] = (
                        judgeability * judgeability
                    )

                if (
                    visible >= self.min_region_visible_probability
                    and visible > absent
                    and visible > unjudgeable
                ):
                    for name in item.get("regions", []):
                        if name in self.region_index:
                            region_target[
                                symptom_index,
                                self.region_index[name],
                            ] = 1.0
                    if region_target[symptom_index].sum() > 0:
                        region_target[symptom_index] /= (
                            region_target[symptom_index].sum()
                        )
                        region_mask[symptom_index] = True
        return (
            images,
            label,
            path,
            target,
            supervision_weight,
            region_target,
            region_mask,
        )
