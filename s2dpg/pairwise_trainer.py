from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .finetune_trainer import evaluate


def evaluate_pairwise(model, loader, device, cfg):
    model.eval()
    logits, base_logits, labels, paths = [], [], [], []
    active_rates, corrections = [], []
    with torch.no_grad():
        for images, target, image_paths in loader:
            output = model(images.to(device, non_blocking=True))
            logits.append(output["logits"].cpu())
            base_logits.append(output["base_logits"].cpu())
            labels.append(target.long())
            paths.extend(image_paths)
            active_rates.append(
                output["active_pairs"].float().mean().cpu()
            )
            corrections.append(
                output["correction"].abs().mean().cpu()
            )
    from .metrics import classification_metrics
    logits = torch.cat(logits)
    base_logits = torch.cat(base_logits)
    labels = torch.cat(labels)
    metrics, matrix = classification_metrics(
        logits, labels, model.num_classes,
        int(cfg["evaluation"]["ece_bins"]),
    )
    base_metrics, _ = classification_metrics(
        base_logits, labels, model.num_classes,
        int(cfg["evaluation"]["ece_bins"]),
    )
    return {
        "logits": logits,
        "base_logits": base_logits,
        "labels": labels,
        "paths": paths,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "matrix": matrix,
        "active_pair_rate": float(torch.stack(active_rates).mean()),
        "correction_abs_mean": float(torch.stack(corrections).mean()),
    }


def train_pairwise(model, train_loader, val_loader, cfg, device):
    section = cfg["pairwise"]
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(section["lr"]),
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(section["epochs"]) * len(train_loader)),
    )
    with torch.no_grad():
        model.correction_strength_raw.fill_(-20.0)
    initial = evaluate_pairwise(model, val_loader, device, cfg)
    best_score = initial["base_metrics"]["macro_f1"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.correction_strength_raw.fill_(
            float(model.cfg["correction_strength_init"])
        )
    stale = 0
    history = []
    for epoch in range(int(section["epochs"])):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "pair": 0.0}
        seen = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            classification = F.cross_entropy(
                output["logits"], labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            pair = model.pairwise_loss(output, labels)
            loss = (
                classification
                + float(section["lambda_pair"]) * pair
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            count = len(labels)
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["classification"] += (
                float(classification.detach()) * count
            )
            totals["pair"] += float(pair.detach()) * count
        validation = evaluate_pairwise(
            model, val_loader, device, cfg
        )
        row = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / seen,
            "classification_loss": totals["classification"] / seen,
            "pair_loss": totals["pair"] / seen,
            "base_val_macro_f1":
                validation["base_metrics"]["macro_f1"],
            "active_pair_rate": validation["active_pair_rate"],
            "correction_abs_mean":
                validation["correction_abs_mean"],
            "strength": float(
                model.correction_strength_raw.sigmoid()
                * float(model.cfg["max_correction"])
            ),
            **validation["metrics"],
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f pair=%.4f base_f1=%.2f "
            "val_acc=%.2f val_f1=%.2f corr=%.4f strength=%.3f"
            % (
                epoch + 1, row["loss"], row["pair_loss"],
                row["base_val_macro_f1"], row["accuracy"],
                row["macro_f1"], row["correction_abs_mean"],
                row["strength"],
            )
        )
        score = row["macro_f1"]
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
        "base_val_macro_f1":
            initial["base_metrics"]["macro_f1"],
        "history": history,
    }
