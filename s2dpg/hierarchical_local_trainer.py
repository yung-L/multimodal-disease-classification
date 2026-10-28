from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .metrics import classification_metrics


def evaluate_hierarchical(model, loader, device, cfg):
    model.eval()
    logits, base_logits, local_logits = [], [], []
    labels, paths = [], []
    gates, null_values, reliabilities = [], [], []
    with torch.no_grad():
        for images, target, image_paths in loader:
            output = model(images.to(device, non_blocking=True))
            logits.append(output["logits"].cpu())
            base_logits.append(output["base_logits"].cpu())
            local_logits.append(output["local_logits"].cpu())
            labels.append(target.long())
            paths.extend(image_paths)
            gates.append(output["fusion_gate"].mean().cpu())
            null_values.append(output["null_attention"].mean().cpu())
            reliabilities.append(
                output["symptom_reliability"].mean().cpu()
            )
    logits = torch.cat(logits)
    base_logits = torch.cat(base_logits)
    local_logits = torch.cat(local_logits)
    labels = torch.cat(labels)
    bins = int(cfg["evaluation"]["ece_bins"])
    metrics, matrix = classification_metrics(
        logits, labels, model.num_classes, bins
    )
    base_metrics, _ = classification_metrics(
        base_logits, labels, model.num_classes, bins
    )
    local_metrics, _ = classification_metrics(
        local_logits, labels, model.num_classes, bins
    )
    return {
        "logits": logits,
        "base_logits": base_logits,
        "local_logits": local_logits,
        "labels": labels,
        "paths": paths,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "local_metrics": local_metrics,
        "matrix": matrix,
        "mean_gate": float(torch.stack(gates).mean()),
        "mean_null_attention": float(
            torch.stack(null_values).mean()
        ),
        "mean_symptom_reliability": float(
            torch.stack(reliabilities).mean()
        ),
    }


def train_hierarchical_local(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
):
    section = cfg["hierarchical_local"]
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

    initial = evaluate_hierarchical(model, val_loader, device, cfg)
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
            "local_classification": 0.0,
            "consistency": 0.0,
            "sparsity": 0.0,
            "gate_regularization": 0.0,
            "distillation": 0.0,
            "gate": 0.0,
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
            local_classification = F.cross_entropy(
                output["local_logits"],
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            consistency = model.local_consistency_loss(output)
            sparsity = model.attention_sparsity_loss(output)
            gate_regularization = model.gate_regularization_loss(output)
            distillation = model.distillation_loss(output)
            anchor = model.prototype_anchor_loss()
            loss = (
                classification
                + float(section["lambda_local_classification"])
                * local_classification
                + float(section["lambda_consistency"]) * consistency
                + float(section["lambda_sparsity"]) * sparsity
                + float(section["lambda_gate_regularization"])
                * gate_regularization
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
            scheduler.step()

            count = len(labels)
            seen += count
            values = {
                "loss": loss,
                "classification": classification,
                "local_classification": local_classification,
                "consistency": consistency,
                "sparsity": sparsity,
                "gate_regularization": gate_regularization,
                "distillation": distillation,
                "gate": output["fusion_gate"].mean(),
            }
            for key, value in values.items():
                totals[key] += float(value.detach()) * count

        validation = evaluate_hierarchical(
            model, val_loader, device, cfg
        )
        metrics = validation["metrics"]
        row = {
            "epoch": epoch + 1,
            **{
                key + "_loss" if key != "gate" else "mean_gate":
                value / seen
                for key, value in totals.items()
            },
            **metrics,
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f cls=%.4f local=%.4f "
            "cons=%.4f sparse=%.4f gate=%.3f "
            "val_acc=%.2f val_f1=%.2f"
            % (
                epoch + 1,
                row["loss_loss"],
                row["classification_loss"],
                row["local_classification_loss"],
                row["consistency_loss"],
                row["sparsity_loss"],
                row["mean_gate"],
                metrics["accuracy"],
                metrics["macro_f1"],
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

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "initial_val_macro_f1": initial["metrics"]["macro_f1"],
        "history": history,
    }
