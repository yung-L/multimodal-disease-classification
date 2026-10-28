from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics


def class_weights(
    labels: torch.Tensor, num_classes: int, power: float
) -> torch.Tensor:
    counts = torch.bincount(
        labels, minlength=num_classes
    ).float().clamp_min(1)
    weights = counts.pow(-power)
    return weights / weights.mean()


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor
) -> float:
    log_temperature = torch.zeros(
        (), device=logits.device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(
        log_temperature.detach().exp().clamp(0.05, 20)
    )


def _metrics(model, regions, labels, cfg):
    output = model(regions)
    metrics, _ = classification_metrics(
        output["logits"].cpu(),
        labels.cpu(),
        model.num_classes,
        int(cfg["evaluation"]["ece_bins"]),
    )
    baseline, _ = classification_metrics(
        output["baseline_logits"].cpu(),
        labels.cpu(),
        model.num_classes,
        int(cfg["evaluation"]["ece_bins"]),
    )
    candidate_recall = float(
        output["candidate_ids"]
        .eq(labels[:, None])
        .any(dim=1)
        .float()
        .mean()
    )
    return output, metrics, baseline, candidate_recall


def _loader(regions, labels, train_cfg, seed):
    return DataLoader(
        TensorDataset(regions.float(), labels.long()),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_model(
    model,
    train_regions: torch.Tensor,
    train_labels: torch.Tensor,
    val_regions: torch.Tensor,
    val_labels: torch.Tensor,
    cfg: dict,
    device: torch.device,
):
    train_cfg = cfg["train"]
    seed = int(cfg["experiment"]["seed"])
    requested_ablation = model.ablation
    model = model.to(device)
    model.initialize_visual_components(
        train_regions[:, 0], train_labels, seed
    )
    loader = _loader(
        train_regions, train_labels, train_cfg, seed
    )
    weights = None
    if train_cfg.get("class_balanced_loss", False):
        weights = class_weights(
            train_labels,
            model.num_classes,
            float(train_cfg["class_weight_power"]),
        ).to(device)
    val_regions = val_regions.to(device)
    val_labels = val_labels.to(device)
    history = []
    global_step = 0

    # Stage 1: train and validate the exact MVPDR base. Symptom parameters
    # cannot influence model selection in this stage.
    model.ablation = "mvpdr_only"
    base_parameters = [
        model.visual_prototypes,
        model.class_text_features,
    ]
    base_optimizer = torch.optim.AdamW(
        base_parameters,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    base_epochs = int(train_cfg["base_epochs"])
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,
        T_max=max(1, base_epochs * len(loader)),
    )
    base_best_score = -1.0
    base_best_epoch = -1
    base_best_state = None
    base_stale = 0
    for epoch in range(base_epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for regions, labels in loader:
            regions = regions.to(device)
            labels = labels.to(device)
            output = model(regions)
            classification = F.cross_entropy(
                output["baseline_logits"], labels, weight=weights
            )
            auxiliary = model.mvpdr_auxiliary_loss(
                output, labels
            )
            diversity = model.diversity_loss()
            loss = (
                classification
                + float(train_cfg["lambda_mvpdr_auxiliary"])
                * auxiliary
                + float(train_cfg["lambda_diversity"])
                * diversity
            )
            base_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                base_parameters, 5.0
            )
            base_optimizer.step()
            base_scheduler.step()
            global_step += 1
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)

        model.eval()
        with torch.no_grad():
            _, metrics, baseline, recall = _metrics(
                model, val_regions, val_labels, cfg
            )
        row = {
            "stage": "mvpdr_base",
            "epoch": epoch + 1,
            "global_step": global_step,
            "loss": total_loss / seen,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "baseline_val_accuracy": baseline["accuracy"],
            "baseline_val_macro_f1": baseline["macro_f1"],
            "candidate_recall": recall,
        }
        history.append(row)
        score = baseline["macro_f1"]
        print(
            "Base %03d loss=%.4f val_acc=%.2f val_f1=%.2f "
            "cand_recall=%.3f"
            % (
                epoch + 1,
                row["loss"],
                baseline["accuracy"],
                score,
                recall,
            )
        )
        if score > base_best_score:
            base_best_score = score
            base_best_epoch = epoch + 1
            base_best_state = copy.deepcopy(model.state_dict())
            base_stale = 0
        else:
            base_stale += 1
        if base_stale >= int(train_cfg["patience"]):
            break
    if base_best_state is None:
        raise RuntimeError("MVPDR base produced no checkpoint")
    model.load_state_dict(base_best_state)

    # A strict baseline ends here.
    if requested_ablation in {"mvpdr_only", "no_symptom_branch"}:
        model.ablation = "mvpdr_only"
        model.eval()
        with torch.no_grad():
            validation_logits = model(val_regions)["logits"]
        temperature = 1.0
        if cfg["evaluation"].get("temperature_scaling", True):
            temperature = fit_temperature(
                validation_logits, val_labels
            )
        return model, {
            "base_best_epoch": base_best_epoch,
            "base_best_val_macro_f1": base_best_score,
            "best_epoch": base_best_epoch,
            "best_val_macro_f1": base_best_score,
            "calibration_temperature": temperature,
            "optimizer_steps": global_step,
            "history": history,
        }

    # Stage 2: freeze the strong base and train only the verifier.
    model.visual_prototypes.requires_grad_(False)
    model.class_text_features.requires_grad_(False)
    model.ablation = requested_ablation
    with torch.no_grad():
        model.atomic_strength_raw.fill_(-20.0)
        model.hyperedge_strength_raw.fill_(-20.0)

    verifier_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    verifier_epochs = int(train_cfg["verifier_epochs"])
    verifier_optimizer = torch.optim.AdamW(
        verifier_parameters,
        lr=float(train_cfg["verifier_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    verifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        verifier_optimizer,
        T_max=max(1, verifier_epochs * len(loader)),
    )

    # Zero-correction checkpoint is the safety floor. The verifier is kept
    # only when validation Macro-F1 improves over the frozen MVPDR base.
    model.eval()
    with torch.no_grad():
        _, initial_metrics, _, _ = _metrics(
            model, val_regions, val_labels, cfg
        )
    best_score = max(
        base_best_score, initial_metrics["macro_f1"]
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    with torch.no_grad():
        model.atomic_strength_raw.fill_(-5.0)
        model.hyperedge_strength_raw.fill_(-5.0)

    for epoch in range(verifier_epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "symptom": 0.0,
            "hyperedge": 0.0,
            "correction": 0.0,
        }
        seen = 0
        for regions, labels in loader:
            regions = regions.to(device)
            labels = labels.to(device)
            output = model(regions)
            classification = F.cross_entropy(
                output["logits"], labels, weight=weights
            )
            symptom = model.symptom_rank_loss(
                output["symptom_logits"], labels
            )
            hyperedge = model.hyperedge_rank_loss(
                output["hyperedge_activation"], labels
            )
            correction = model.correction_margin_loss(
                output, labels
            )
            adapter_regularization = sum(
                parameter.pow(2).mean()
                for parameter in model.symptom_adapter.parameters()
            )
            loss = (
                classification
                + float(train_cfg["lambda_symptom"]) * symptom
                + float(train_cfg["lambda_hyperedge"]) * hyperedge
                + float(train_cfg["lambda_correction"]) * correction
                + float(train_cfg["lambda_symptom_adapter"])
                * adapter_regularization
            )
            verifier_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                verifier_parameters, 5.0
            )
            verifier_optimizer.step()
            verifier_scheduler.step()
            global_step += 1
            count = len(labels)
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["classification"] += (
                float(classification.detach()) * count
            )
            totals["symptom"] += float(symptom.detach()) * count
            totals["hyperedge"] += float(hyperedge.detach()) * count
            totals["correction"] += (
                float(correction.detach()) * count
            )

        model.eval()
        with torch.no_grad():
            val_output, metrics, baseline, recall = _metrics(
                model, val_regions, val_labels, cfg
            )
        row = {
            "stage": "symptom_verifier",
            "epoch": epoch + 1,
            "global_step": global_step,
            "loss": totals["loss"] / seen,
            "classification_loss":
                totals["classification"] / seen,
            "symptom_loss": totals["symptom"] / seen,
            "hyperedge_loss": totals["hyperedge"] / seen,
            "correction_loss": totals["correction"] / seen,
            "baseline_val_accuracy": baseline["accuracy"],
            "baseline_val_macro_f1": baseline["macro_f1"],
            "candidate_recall": recall,
            "atomic_strength": float(
                val_output["atomic_strength"]
            ),
            "hyperedge_strength": float(
                val_output["hyperedge_strength"]
            ),
        }
        row.update(metrics)
        history.append(row)
        score = metrics["macro_f1"]
        print(
            "Verify %03d loss=%.4f base_f1=%.2f "
            "val_acc=%.2f val_f1=%.2f atomic=%.3f edge=%.3f"
            % (
                epoch + 1,
                row["loss"],
                baseline["macro_f1"],
                metrics["accuracy"],
                score,
                row["atomic_strength"],
                row["hyperedge_strength"],
            )
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(train_cfg["verifier_patience"]):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logits = model(val_regions)["logits"]
    temperature = 1.0
    if cfg["evaluation"].get("temperature_scaling", True):
        temperature = fit_temperature(
            validation_logits, val_labels
        )
    return model, {
        "base_best_epoch": base_best_epoch,
        "base_best_val_macro_f1": base_best_score,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "calibration_temperature": temperature,
        "optimizer_steps": global_step,
        "history": history,
    }
