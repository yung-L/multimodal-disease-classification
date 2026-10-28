from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import normalize
from .dino_probe import evaluate_probe


def build_symptom_similarity(graph: dict, hyperedge_weight: float):
    positive = torch.tensor(
        graph["positive_matrix"], dtype=torch.float32
    )
    idf = torch.tensor([
        float(item.get("idf", 1.0)) for item in graph["symptoms"]
    ])
    weighted = positive * idf[None]
    intersection = weighted @ positive.t()
    mass = weighted.sum(dim=1)
    union = (
        mass[:, None] + mass[None] - intersection
    ).clamp_min(1e-6)
    atomic = intersection / union

    edge_sets = [[] for _ in graph["classnames"]]
    for edge in graph.get("composite_hyperedges", []):
        components = set(edge["atomic_symptom_indices"])
        if len(components) >= 2:
            edge_sets[int(edge["class_id"])].append(components)
    hyperedge = torch.zeros_like(atomic)
    for first in range(len(edge_sets)):
        for second in range(first + 1, len(edge_sets)):
            overlaps = [
                len(a & b) / max(len(a | b), 1)
                for a in edge_sets[first]
                for b in edge_sets[second]
            ]
            if overlaps:
                value = max(overlaps)
                hyperedge[first, second] = value
                hyperedge[second, first] = value

    similarity = (
        (1.0 - hyperedge_weight) * atomic
        + hyperedge_weight * hyperedge
    ).clamp(0, 1)
    similarity.fill_diagonal_(0)
    return similarity, atomic, hyperedge


def hard_negative_mask(
    logits,
    labels,
    symptom_similarity,
    visual_topk,
    symptom_topk,
):
    batch, classes = logits.shape
    candidate_logits = logits.detach().clone()
    candidate_logits.scatter_(1, labels[:, None], -1e4)
    visual_ids = candidate_logits.topk(
        min(visual_topk, classes - 1), dim=1
    ).indices

    symptom_rows = symptom_similarity[labels].clone()
    symptom_rows.scatter_(1, labels[:, None], -1)
    symptom_ids = symptom_rows.topk(
        min(symptom_topk, classes - 1), dim=1
    ).indices

    visual_mask = torch.zeros(
        batch, classes, dtype=torch.bool, device=logits.device
    )
    symptom_mask = torch.zeros_like(visual_mask)
    visual_mask.scatter_(1, visual_ids, True)
    symptom_mask.scatter_(1, symptom_ids, True)
    selected = visual_mask & symptom_mask

    # If no symptom-neighbor is currently visually confusing, retain the
    # strongest visual competitor so every sample contributes a margin.
    empty = ~selected.any(dim=1)
    if empty.any():
        selected[empty, visual_ids[empty, 0]] = True
    selected.scatter_(1, labels[:, None], False)
    return selected, visual_ids, symptom_ids


def margin_logits(
    logits,
    labels,
    similarity,
    mode,
    base_margin,
    similarity_scale,
    visual_topk,
    symptom_topk,
):
    symptom_selected, visual_ids, symptom_ids = hard_negative_mask(
        logits,
        labels,
        similarity,
        visual_topk,
        symptom_topk,
    )
    if mode == "plain_margin":
        selected = torch.zeros_like(symptom_selected)
        selected.scatter_(1, visual_ids, True)
        selected.scatter_(1, labels[:, None], False)
        margins = torch.full_like(logits, base_margin)
    elif mode == "symptom_margin":
        selected = symptom_selected
        class_similarity = similarity[labels]
        margins = base_margin * (
            1.0 + similarity_scale * class_similarity
        )
    else:
        raise ValueError("Unsupported margin mode: %s" % mode)
    adjusted = logits + margins * selected.float()
    return adjusted, selected, visual_ids, symptom_ids


def train_margin_probe(
    probe,
    cache,
    graph,
    cfg,
    device,
    mode,
):
    section = cfg["margin"]
    probe = probe.to(device)
    teacher = copy.deepcopy(probe).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    initial_weight = normalize(
        probe.weight.detach().clone()
    ).to(device)
    similarity, atomic, hyperedge = build_symptom_similarity(
        graph, float(section["hyperedge_weight"])
    )
    similarity = similarity.to(device)

    train_features = cache["train"]["features"].float()
    train_labels = cache["train"]["labels"].long()
    val_features = cache["val"]["features"].float()
    val_labels = cache["val"]["labels"].long()
    class_counts = torch.bincount(
        train_labels, minlength=len(graph["classnames"])
    ).float().to(device)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=int(section["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(section["lr"]),
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(section["epochs"])
    )

    _, initial_metrics, _ = evaluate_probe(
        probe,
        val_features,
        val_labels,
        len(graph["classnames"]),
        int(cfg["evaluation"]["ece_bins"]),
    )
    best_score = initial_metrics["macro_f1"]
    best_epoch = 0
    best_state = copy.deepcopy(probe.state_dict())
    stale = 0
    history = []
    for epoch in range(int(section["epochs"])):
        probe.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "margin": 0.0,
            "distillation": 0.0,
            "anchor": 0.0,
            "selected_similarity": 0.0,
            "selected_count": 0.0,
        }
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = probe(features)
            balanced_logits = (
                logits
                + float(section["balance_strength"])
                * class_counts.clamp_min(1).log()[None]
            )
            classification = F.cross_entropy(
                balanced_logits,
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            adjusted, selected, _, _ = margin_logits(
                balanced_logits,
                labels,
                similarity,
                mode,
                float(section["base_margin"]),
                float(section["similarity_scale"]),
                int(section["visual_topk"]),
                int(section["symptom_topk"]),
            )
            margin_loss = F.cross_entropy(
                adjusted,
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            with torch.no_grad():
                teacher_logits = teacher(features)
            temperature = float(section["distill_temperature"])
            distillation = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * temperature * temperature
            anchor = (
                1.0
                - F.cosine_similarity(
                    normalize(probe.weight),
                    initial_weight,
                    dim=1,
                )
            ).mean()
            loss = (
                float(section["lambda_classification"])
                * classification
                + float(section["lambda_margin"]) * margin_loss
                + float(section["lambda_distillation"])
                * distillation
                + float(section["lambda_anchor"]) * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                probe.parameters(), float(section["gradient_clip"])
            )
            optimizer.step()

            count = len(labels)
            seen += count
            selected_similarity = similarity[labels][selected]
            totals["loss"] += float(loss.detach()) * count
            totals["classification"] += (
                float(classification.detach()) * count
            )
            totals["margin"] += float(margin_loss.detach()) * count
            totals["distillation"] += (
                float(distillation.detach()) * count
            )
            totals["anchor"] += float(anchor.detach()) * count
            totals["selected_similarity"] += float(
                selected_similarity.sum().detach()
            )
            totals["selected_count"] += int(selected.sum())
        scheduler.step()

        _, metrics, _ = evaluate_probe(
            probe,
            val_features,
            val_labels,
            len(graph["classnames"]),
            int(cfg["evaluation"]["ece_bins"]),
        )
        row = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / seen,
            "classification_loss":
                totals["classification"] / seen,
            "margin_loss": totals["margin"] / seen,
            "distillation_loss":
                totals["distillation"] / seen,
            "anchor_loss": totals["anchor"] / seen,
            "mean_selected_similarity":
                totals["selected_similarity"]
                / max(totals["selected_count"], 1),
            "mean_selected_negatives":
                totals["selected_count"] / seen,
            **metrics,
        }
        history.append(row)
        print(
            "Margin epoch %03d loss=%.4f cls=%.4f margin=%.4f "
            "sim=%.3f neg=%.2f val_acc=%.2f val_f1=%.2f"
            % (
                epoch + 1,
                row["loss"],
                row["classification_loss"],
                row["margin_loss"],
                row["mean_selected_similarity"],
                row["mean_selected_negatives"],
                metrics["accuracy"],
                metrics["macro_f1"],
            )
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(probe.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(section["patience"]):
            break

    probe.load_state_dict(best_state)
    return probe, {
        "mode": mode,
        "best_epoch": best_epoch,
        "initial_val_macro_f1": initial_metrics["macro_f1"],
        "best_val_macro_f1": best_score,
        "history": history,
        "similarity_diagnostics": {
            "mean": float(similarity.mean()),
            "max": float(similarity.max()),
            "atomic_mean": float(atomic.mean()),
            "hyperedge_mean": float(hyperedge.mean()),
        },
    }
