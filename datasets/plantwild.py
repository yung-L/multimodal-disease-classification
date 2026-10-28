from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def _image_root(root: Path) -> Path:
    candidates = (root / "images", root / "plantwild" / "images", root)
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate PlantWild class folders under %s" % root)


def _resolve_image(raw_path: str, image_root: Path, classname: str) -> Path:
    original = Path(raw_path).expanduser()
    if original.is_file():
        return original.resolve()
    portable = PurePosixPath(str(raw_path).replace("\\", "/"))
    parts = list(portable.parts)
    candidates = []
    if classname in parts:
        index = len(parts) - 1 - parts[::-1].index(classname)
        candidates.append(image_root.joinpath(*parts[index:]))
    candidates.extend([image_root / classname / portable.name, image_root / portable.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot remap split image %r (class=%r) below %s"
        % (raw_path, classname, image_root)
    )


@DATASET_REGISTRY.register()
class PlantWild(DatasetBase):
    """PlantWild using the exact fixed split shared by all project baselines."""

    dataset_dir = ""

    def __init__(self, cfg):
        root = Path(os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)))
        image_root = _image_root(root)
        split_file = Path(
            os.environ.get("PLANTWILD_SPLIT_FILE", root / "plantwild_seed1.json")
        ).expanduser().resolve()
        if not split_file.is_file():
            raise FileNotFoundError(
                "%s does not exist. Set PLANTWILD_SPLIT_FILE to the fixed JSON split."
                % split_file
            )

        with split_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = {"train", "val", "test"}
        if set(payload) != expected:
            raise ValueError("Split keys must be exactly %s, got %s" % (expected, set(payload)))

        class_by_label = {}
        seen = set()
        converted = {}
        for split_name in ("train", "val", "test"):
            rows = []
            for item in payload[split_name]:
                label = int(item["label"])
                classname = str(item["classname"])
                previous = class_by_label.setdefault(label, classname)
                if previous != classname:
                    raise ValueError(
                        "Label %d maps to both %r and %r" % (label, previous, classname)
                    )
                impath = _resolve_image(item["path"], image_root, classname)
                key = str(impath).lower()
                if key in seen:
                    raise ValueError("Image occurs in more than one split: %s" % impath)
                seen.add(key)
                rows.append(Datum(impath=str(impath), label=label, classname=classname))
            converted[split_name] = rows

        labels = sorted(class_by_label)
        if labels != list(range(len(labels))):
            raise ValueError("Class labels must be contiguous from zero, got %s" % labels)
        if len(labels) != 89:
            raise ValueError("PlantWild experiment expects 89 classes, got %d" % len(labels))

        print("PlantWild fixed split: %s" % split_file)
        print("PlantWild image root: %s" % image_root)
        print(
            "PlantWild sizes: train=%d val=%d test=%d classes=%d"
            % (len(converted["train"]), len(converted["val"]),
               len(converted["test"]), len(labels))
        )
        super().__init__(
            train_x=converted["train"], val=converted["val"], test=converted["test"]
        )
