from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import classification_metrics


def evaluate(model, loader, device, cfg):
    model.eval()
    logits, labels, paths = [], [], []
    symptom_values = []
    latent_corrections = []
    hierarchy_atomic_corrections = []
    hierarchy_hyperedge_corrections = []
    hierarchy_monotonic_accuracy = []
    hierarchy_margin_gain = []
    hierarchy_visibility_coverage = []
    hierarchy_atomic_monotonic_accuracy = []
    hierarchy_atomic_margin_gain = []
    hierarchy_atomic_visibility_coverage = []
    hierarchy_composite_monotonic_accuracy = []
    hierarchy_composite_margin_gain = []
    hierarchy_composite_visibility_coverage = []
    with torch.no_grad():
        for images, target, image_paths in loader:
            target = target.long().to(device, non_blocking=True)
            output = model(images.to(device, non_blocking=True))
            logits.append(output["logits"].cpu())
            labels.append(target.cpu())
            paths.extend(image_paths)
            symptom_values.append(
                output["symptom_scores"].sigmoid().mean().cpu()
            )
            latent_corrections.append(
                output["latent_correction"].abs().mean().cpu()
            )
            hierarchy_atomic_corrections.append(
                output["hierarchy_atomic_correction"]
                .abs().mean().cpu()
            )
            hierarchy_hyperedge_corrections.append(
                output["hierarchy_hyperedge_correction"]
                .abs().mean().cpu()
            )
            _, _, hierarchy_diag = model.hierarchy_losses(
                output, target
            )
            hierarchy_monotonic_accuracy.append(
                hierarchy_diag["monotonic_accuracy"].cpu()
            )
            hierarchy_margin_gain.append(
                hierarchy_diag["margin_gain"].cpu()
            )
            hierarchy_visibility_coverage.append(
                hierarchy_diag["visibility_coverage"].cpu()
            )
            hierarchy_atomic_monotonic_accuracy.append(
                hierarchy_diag["atomic_monotonic_accuracy"].cpu()
            )
            hierarchy_atomic_margin_gain.append(
                hierarchy_diag["atomic_margin_gain"].cpu()
            )
            hierarchy_atomic_visibility_coverage.append(
                hierarchy_diag["atomic_visibility_coverage"].cpu()
            )
            hierarchy_composite_monotonic_accuracy.append(
                hierarchy_diag["composite_monotonic_accuracy"].cpu()
            )
            hierarchy_composite_margin_gain.append(
                hierarchy_diag["composite_margin_gain"].cpu()
            )
            hierarchy_composite_visibility_coverage.append(
                hierarchy_diag[
                    "composite_visibility_coverage"
                ].cpu()
            )
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    metrics, matrix = classification_metrics(
        logits,
        labels,
        model.num_classes,
        int(cfg["evaluation"]["ece_bins"]),
    )
    return {
        "logits": logits,
        "labels": labels,
        "paths": paths,
        "metrics": metrics,
        "matrix": matrix,
        "symptom_probability_mean": float(
            torch.stack(symptom_values).mean()
        ),
        "latent_correction_abs_mean": float(
            torch.stack(latent_corrections).mean()
        ),
        "hierarchy_atomic_correction_abs_mean": float(
            torch.stack(hierarchy_atomic_corrections).mean()
        ),
        "hierarchy_hyperedge_correction_abs_mean": float(
            torch.stack(hierarchy_hyperedge_corrections).mean()
        ),
        "hierarchy_monotonic_accuracy": float(
            torch.stack(hierarchy_monotonic_accuracy).mean()
        ),
        "hierarchy_margin_gain": float(
            torch.stack(hierarchy_margin_gain).mean()
        ),
        "hierarchy_visibility_coverage": float(
            torch.stack(hierarchy_visibility_coverage).mean()
        ),
        "hierarchy_atomic_monotonic_accuracy": float(
            torch.stack(
                hierarchy_atomic_monotonic_accuracy
            ).mean()
        ),
        "hierarchy_atomic_margin_gain": float(
            torch.stack(hierarchy_atomic_margin_gain).mean()
        ),
        "hierarchy_atomic_visibility_coverage": float(
            torch.stack(
                hierarchy_atomic_visibility_coverage
            ).mean()
        ),
        "hierarchy_composite_monotonic_accuracy": float(
            torch.stack(
                hierarchy_composite_monotonic_accuracy
            ).mean()
        ),
        "hierarchy_composite_margin_gain": float(
            torch.stack(
                hierarchy_composite_margin_gain
            ).mean()
        ),
        "hierarchy_composite_visibility_coverage": float(
            torch.stack(
                hierarchy_composite_visibility_coverage
            ).mean()
        ),
    }


def train_finetune(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
):
    section = cfg["finetune"]
    model = model.to(device)
    backbone, head = model.trainable_groups()
    parameter_groups = []
    if backbone:
        parameter_groups.append({
            "params": backbone,
            "lr": float(section["backbone_lr"]),
        })
    parameter_groups.append({
        "params": head,
        "lr": float(section["head_lr"]),
    })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(section["epochs"]) * len(train_loader)),
    )
    best_score = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    for epoch in range(int(section["epochs"])):
        model.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "atomic": 0.0,
            "hyperedge": 0.0,
            "graph_margin": 0.0,
            "hierarchy": 0.0,
            "monotonic": 0.0,
            "distillation": 0.0,
        }
        seen = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            classification = F.cross_entropy(
                output["logits"],
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            atomic = model.atomic_loss(
                output["symptom_scores"], labels, output["logits"]
            )
            hyperedge = model.hyperedge_loss(
                output["hyperedge_scores"], labels, output["logits"]
            )
            graph_margin = model.graph_margin_loss(
                output["logits"], labels
            )
            hierarchy, monotonic, hierarchy_diag = (
                model.hierarchy_losses(output, labels)
            )
            distillation = model.distillation_loss(output)
            anchor = model.prototype_anchor_loss()
            loss = (
                classification
                + float(section["lambda_atomic"]) * atomic
                + float(section["lambda_hyperedge"]) * hyperedge
                + float(section["lambda_graph_margin"])
                * graph_margin
                + float(section.get("lambda_hierarchy", 0.0))
                * hierarchy
                + float(section.get("lambda_monotonic", 0.0))
                * monotonic
                + float(section["lambda_distillation"]) * distillation
                + float(section["lambda_prototype_anchor"]) * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            count = len(labels)
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["classification"] += (
                float(classification.detach()) * count
            )
            totals["atomic"] += float(atomic.detach()) * count
            totals["hyperedge"] += float(hyperedge.detach()) * count
            totals["graph_margin"] += (
                float(graph_margin.detach()) * count
            )
            totals["hierarchy"] += (
                float(hierarchy.detach()) * count
            )
            totals["monotonic"] += (
                float(monotonic.detach()) * count
            )
            totals["distillation"] += (
                float(distillation.detach()) * count
            )

        validation = evaluate(model, val_loader, device, cfg)
        metrics = validation["metrics"]
        row = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / seen,
            "classification_loss": totals["classification"] / seen,
            "atomic_loss": totals["atomic"] / seen,
            "hyperedge_loss": totals["hyperedge"] / seen,
            "graph_margin_loss":
                totals["graph_margin"] / seen,
            "hierarchy_loss": totals["hierarchy"] / seen,
            "monotonic_loss": totals["monotonic"] / seen,
            "distillation_loss": totals["distillation"] / seen,
            "val_monotonic_accuracy":
                validation["hierarchy_monotonic_accuracy"],
            "val_margin_gain":
                validation["hierarchy_margin_gain"],
            "val_visibility_coverage":
                validation["hierarchy_visibility_coverage"],
            "val_atomic_monotonic_accuracy":
                validation[
                    "hierarchy_atomic_monotonic_accuracy"
                ],
            "val_atomic_margin_gain":
                validation["hierarchy_atomic_margin_gain"],
            "val_atomic_visibility_coverage":
                validation[
                    "hierarchy_atomic_visibility_coverage"
                ],
            "val_composite_monotonic_accuracy":
                validation[
                    "hierarchy_composite_monotonic_accuracy"
                ],
            "val_composite_margin_gain":
                validation["hierarchy_composite_margin_gain"],
            "val_composite_visibility_coverage":
                validation[
                    "hierarchy_composite_visibility_coverage"
                ],
            **metrics,
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f cls=%.4f graph=%.4f "
            "atomic=%.4f edge=%.4f hier=%.4f mono=%.4f "
            "distill=%.4f val_acc=%.2f val_f1=%.2f "
            "cma=%.3f gain=%.4f coverage=%.3f "
            "atomic=(%.3f,%.4f,%.3f) "
            "composite=(%.3f,%.4f,%.3f)"
            % (
                epoch + 1,
                row["loss"],
                row["classification_loss"],
                row["graph_margin_loss"],
                row["atomic_loss"],
                row["hyperedge_loss"],
                row["hierarchy_loss"],
                row["monotonic_loss"],
                row["distillation_loss"],
                metrics["accuracy"],
                metrics["macro_f1"],
                row["val_monotonic_accuracy"],
                row["val_margin_gain"],
                row["val_visibility_coverage"],
                row["val_atomic_monotonic_accuracy"],
                row["val_atomic_margin_gain"],
                row["val_atomic_visibility_coverage"],
                row["val_composite_monotonic_accuracy"],
                row["val_composite_margin_gain"],
                row["val_composite_visibility_coverage"],
            )
        )
        score = metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(section["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "history": history,
    }
