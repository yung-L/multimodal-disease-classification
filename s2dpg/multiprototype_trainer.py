from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from .metrics import classification_metrics


def balanced_cross_entropy(
    logits,
    labels,
    class_counts,
    strength,
    label_smoothing,
):
    adjustment = strength * class_counts.clamp_min(1).log()
    return F.cross_entropy(
        logits + adjustment[None],
        labels,
        label_smoothing=label_smoothing,
    )


def evaluate_multiprototype(model, loader, device, cfg):
    model.eval()
    output_values = {
        "logits": [],
        "base_logits": [],
        "visual_logits": [],
        "semantic_logits": [],
    }
    labels, paths = [], []
    visual_gates, semantic_gates = [], []
    visual_entropies, semantic_entropies = [], []
    with torch.no_grad():
        for images, target, image_paths in loader:
            output = model(images.to(device, non_blocking=True))
            for key in output_values:
                output_values[key].append(output[key].cpu())
            labels.append(target.long())
            paths.extend(image_paths)
            visual_gates.append(output["visual_gate"].mean().cpu())
            semantic_gates.append(output["semantic_gate"].mean().cpu())
            visual_entropies.append(
                -(
                    output["visual_assignment"]
                    * output["visual_assignment"].clamp_min(1e-8).log()
                ).sum(dim=2).mean().cpu()
            )
            semantic_entropies.append(
                -(
                    output["semantic_assignment"]
                    * output["semantic_assignment"].clamp_min(1e-8).log()
                ).sum(dim=2).mean().cpu()
            )
    output_values = {
        key: torch.cat(values) for key, values in output_values.items()
    }
    labels = torch.cat(labels)
    bins = int(cfg["evaluation"]["ece_bins"])
    metrics = {}
    matrices = {}
    for key, logits in output_values.items():
        metrics[key], matrices[key] = classification_metrics(
            logits, labels, model.num_classes, bins
        )
    return {
        **output_values,
        "labels": labels,
        "paths": paths,
        "metrics": metrics["logits"],
        "base_metrics": metrics["base_logits"],
        "visual_metrics": metrics["visual_logits"],
        "semantic_metrics": metrics["semantic_logits"],
        "matrix": matrices["logits"],
        "mean_visual_gate": float(torch.stack(visual_gates).mean()),
        "mean_semantic_gate": float(torch.stack(semantic_gates).mean()),
        "mean_visual_route_entropy": float(
            torch.stack(visual_entropies).mean()
        ),
        "mean_semantic_route_entropy": float(
            torch.stack(semantic_entropies).mean()
        ),
    }


def train_multiprototype(
    model,
    train_loader,
    val_loader,
    class_counts,
    cfg,
    device,
):
    section = cfg["multiprototype"]
    model = model.to(device)
    backbone, base_head, novel = model.trainable_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": 0.0},
            {"params": base_head, "lr": 0.0},
            {
                "params": novel,
                "lr": float(section["prototype_lr"]),
            },
        ],
        weight_decay=float(section["weight_decay"]),
    )
    class_counts = class_counts.to(device).float()
    warmup_epochs = int(section["prototype_warmup_epochs"])
    total_steps = int(section["epochs"]) * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    joint_steps = max(1, total_steps - warmup_steps)
    global_step = 0

    initial = evaluate_multiprototype(
        model, val_loader, device, cfg
    )
    best_score = initial["metrics"]["macro_f1"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(int(section["epochs"])):
        model.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "visual": 0.0,
            "semantic": 0.0,
            "diversity": 0.0,
            "alignment": 0.0,
            "route_entropy": 0.0,
            "distillation": 0.0,
            "visual_gate": 0.0,
            "semantic_gate": 0.0,
        }
        seen = 0
        for images, labels, _ in train_loader:
            if global_step < warmup_steps:
                joint_factor = 0.0
            else:
                progress = min(
                    1.0,
                    (global_step - warmup_steps) / joint_steps,
                )
                joint_factor = 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            optimizer.param_groups[0]["lr"] = (
                float(section["backbone_lr"]) * joint_factor
            )
            optimizer.param_groups[1]["lr"] = (
                float(section["base_head_lr"]) * joint_factor
            )
            optimizer.param_groups[2]["lr"] = (
                float(section["prototype_lr"])
                * (0.20 + 0.80 * (
                    joint_factor if global_step >= warmup_steps else 1.0
                ))
            )

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            classification = balanced_cross_entropy(
                output["logits"],
                labels,
                class_counts,
                float(section["balance_strength"]),
                float(section["label_smoothing"]),
            )
            zero = classification.new_zeros(())
            visual = zero
            semantic = zero
            if model.multiprototype_mode in {"visual", "full"}:
                visual = balanced_cross_entropy(
                    output["visual_logits"],
                    labels,
                    class_counts,
                    float(section["balance_strength"]),
                    float(section["label_smoothing"]),
                )
            if model.multiprototype_mode in {"semantic", "full"}:
                semantic = balanced_cross_entropy(
                    output["semantic_logits"],
                    labels,
                    class_counts,
                    float(section["balance_strength"]),
                    float(section["label_smoothing"]),
                )
            diversity = (
                model.visual_diversity_loss()
                if model.multiprototype_mode in {"visual", "full"}
                else zero
            )
            alignment = (
                model.distribution_alignment_loss()
                if model.multiprototype_mode == "full"
                else zero
            )
            route_entropy = model.route_entropy_loss(output)
            distillation = model.distillation_loss(output)
            anchor = model.prototype_anchor_loss()
            loss = (
                classification
                + float(section["lambda_visual"]) * visual
                + float(section["lambda_semantic"]) * semantic
                + float(section["lambda_diversity"]) * diversity
                + float(section["lambda_alignment"]) * alignment
                + float(section["lambda_route_entropy"]) * route_entropy
                + float(section["lambda_distillation"]) * distillation
                + float(section["lambda_prototype_anchor"]) * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad],
                float(section["gradient_clip"]),
            )
            optimizer.step()
            global_step += 1

            count = len(labels)
            seen += count
            values = {
                "loss": loss,
                "classification": classification,
                "visual": visual,
                "semantic": semantic,
                "diversity": diversity,
                "alignment": alignment,
                "route_entropy": route_entropy,
                "distillation": distillation,
                "visual_gate": output["visual_gate"].mean(),
                "semantic_gate": output["semantic_gate"].mean(),
            }
            for key, value in values.items():
                totals[key] += float(value.detach()) * count

        validation = evaluate_multiprototype(
            model, val_loader, device, cfg
        )
        metrics = validation["metrics"]
        row = {
            "epoch": epoch + 1,
            **{
                key + "_loss"
                if key not in {"visual_gate", "semantic_gate"}
                else "mean_" + key:
                value / seen
                for key, value in totals.items()
            },
            **metrics,
            "base_macro_f1":
                validation["base_metrics"]["macro_f1"],
            "visual_macro_f1":
                validation["visual_metrics"]["macro_f1"],
            "semantic_macro_f1":
                validation["semantic_metrics"]["macro_f1"],
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f cls=%.4f vis=%.4f sem=%.4f "
            "gate=[%.3f,%.3f] val_acc=%.2f val_f1=%.2f "
            "base_f1=%.2f"
            % (
                epoch + 1,
                row["loss_loss"],
                row["classification_loss"],
                row["visual_loss"],
                row["semantic_loss"],
                row["mean_visual_gate"],
                row["mean_semantic_gate"],
                metrics["accuracy"],
                metrics["macro_f1"],
                row["base_macro_f1"],
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
        if (
            epoch + 1 >= warmup_epochs
            and stale >= int(section["patience"])
        ):
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "initial_val_macro_f1": initial["metrics"]["macro_f1"],
        "history": history,
    }

