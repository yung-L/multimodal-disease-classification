from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .finetune_trainer import evaluate


def train_vlm_distill(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    mode,
):
    section = cfg["vlm_distill"]
    model = model.to(device)
    backbone, head = model.trainable_groups()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": float(section["backbone_lr"]),
            },
            {
                "params": head,
                "lr": float(section["head_lr"]),
            },
        ],
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(section["epochs"]) * len(train_loader)),
    )
    initial = evaluate(model, val_loader, device, cfg)
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
            "pseudo_atomic": 0.0,
            "pseudo_region": 0.0,
            "pseudo_edge": 0.0,
            "distillation": 0.0,
        }
        seen = 0
        for batch in train_loader:
            (
                images,
                labels,
                _,
                targets,
                supervision_weight,
                region_targets,
                region_mask,
            ) = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = targets.to(device)
            supervision_weight = supervision_weight.to(device)
            region_targets = region_targets.to(device)
            region_mask = region_mask.to(device)
            output = model(images)
            classification = F.cross_entropy(
                output["logits"],
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            pseudo_atomic, pseudo_region, pseudo_edge = (
                model.pseudo_symptom_losses(
                    output,
                    targets,
                    supervision_weight,
                    region_targets,
                    region_mask,
                )
            )
            if mode == "atomic":
                pseudo_region = pseudo_region * 0
                pseudo_edge = pseudo_edge * 0
            elif mode == "region":
                pseudo_edge = pseudo_edge * 0
            distillation = model.distillation_loss(output)
            anchor = model.prototype_anchor_loss()
            loss = (
                classification
                + float(section["lambda_atomic"]) * pseudo_atomic
                + float(section["lambda_region"]) * pseudo_region
                + float(section["lambda_hyperedge"]) * pseudo_edge
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
            for key, value in [
                ("loss", loss),
                ("classification", classification),
                ("pseudo_atomic", pseudo_atomic),
                ("pseudo_region", pseudo_region),
                ("pseudo_edge", pseudo_edge),
                ("distillation", distillation),
            ]:
                totals[key] += float(value.detach()) * count

        validation = evaluate(model, val_loader, device, cfg)
        metrics = validation["metrics"]
        row = {
            "epoch": epoch + 1,
            **{
                key + "_loss": value / seen
                for key, value in totals.items()
            },
            **metrics,
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f cls=%.4f symptom=%.4f "
            "region=%.4f edge=%.4f val_acc=%.2f val_f1=%.2f"
            % (
                epoch + 1,
                row["loss_loss"],
                row["classification_loss"],
                row["pseudo_atomic_loss"],
                row["pseudo_region_loss"],
                row["pseudo_edge_loss"],
                metrics["accuracy"],
                metrics["macro_f1"],
            )
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(section["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "initial_val_macro_f1": initial["metrics"]["macro_f1"],
        "history": history,
    }
