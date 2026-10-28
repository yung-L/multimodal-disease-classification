from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import normalize
from .dino_probe import evaluate_probe


def build_symptom_topology(graph: dict, atomic_weight: float, edge_weight: float):
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
        mass[:, None] + mass[None, :] - intersection
    ).clamp_min(1e-6)
    atomic = intersection / union

    edge_sets = [[] for _ in range(len(graph["classnames"]))]
    for edge in graph.get("composite_hyperedges", []):
        values = set(edge["atomic_symptom_indices"])
        if len(values) >= 2:
            edge_sets[int(edge["class_id"])].append(values)
    edge_similarity = torch.zeros_like(atomic)
    for first in range(len(edge_sets)):
        for second in range(first + 1, len(edge_sets)):
            values = [
                len(a & b) / max(len(a | b), 1)
                for a in edge_sets[first]
                for b in edge_sets[second]
            ]
            if values:
                edge_similarity[first, second] = max(values)
                edge_similarity[second, first] = max(values)

    topology = (
        atomic_weight * atomic + edge_weight * edge_similarity
    ) / max(atomic_weight + edge_weight, 1e-6)
    topology.fill_diagonal_(0)
    return topology.clamp(0, 1), atomic, edge_similarity


def topology_distribution(topology, temperature, neighbor_topk):
    count = len(topology)
    mask = torch.eye(count, dtype=torch.bool, device=topology.device)
    logits = topology / temperature
    logits = logits.masked_fill(mask, -1e4)
    if 0 < neighbor_topk < count - 1:
        keep = logits.topk(neighbor_topk, dim=1).indices
        sparse_mask = torch.ones_like(mask)
        sparse_mask.scatter_(1, keep, False)
        sparse_mask |= mask
        logits = logits.masked_fill(sparse_mask, -1e4)
    return F.softmax(logits, dim=1)


def prototype_topology_loss(
    probe,
    target_distribution,
    temperature,
    neighbor_topk,
):
    prototypes = normalize(probe.weight)
    similarity = prototypes @ prototypes.t()
    count = len(similarity)
    mask = torch.eye(
        count, dtype=torch.bool, device=similarity.device
    )
    logits = similarity / temperature
    logits = logits.masked_fill(mask, -1e4)
    if 0 < neighbor_topk < count - 1:
        target_keep = target_distribution.topk(
            neighbor_topk, dim=1
        ).indices
        sparse_mask = torch.ones_like(mask)
        sparse_mask.scatter_(1, target_keep, False)
        sparse_mask |= mask
        logits = logits.masked_fill(sparse_mask, -1e4)
    log_probability = F.log_softmax(logits, dim=1)
    return F.kl_div(
        log_probability,
        target_distribution,
        reduction="batchmean",
    )


def prototype_anchor_loss(probe, initial_weight):
    return (
        1.0
        - F.cosine_similarity(
            normalize(probe.weight),
            normalize(initial_weight),
            dim=1,
        )
    ).mean()


def train_topology_probe(
    probe,
    cache,
    graph,
    cfg,
    device,
    mode,
):
    if mode not in {"anchor_only", "topology"}:
        raise ValueError("Unsupported topology mode: %s" % mode)
    section = cfg["topology"]
    probe = probe.to(device)
    initial_weight = normalize(
        probe.weight.detach().clone()
    ).to(device)
    topology, atomic, edge = build_symptom_topology(
        graph,
        float(section["atomic_weight"]),
        float(section["hyperedge_weight"]),
    )
    topology = topology.to(device)
    target_distribution = topology_distribution(
        topology,
        float(section["target_temperature"]),
        int(section["neighbor_topk"]),
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(section["lr"]),
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(section["epochs"])
    )
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
            "topology": 0.0,
            "anchor": 0.0,
        }
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = probe(features)
            adjusted = (
                logits
                + float(section["balance_strength"])
                * class_counts.clamp_min(1).log()[None]
            )
            classification = F.cross_entropy(
                adjusted,
                labels,
                label_smoothing=float(section["label_smoothing"]),
            )
            relation = (
                prototype_topology_loss(
                    probe,
                    target_distribution,
                    float(section["prototype_temperature"]),
                    int(section["neighbor_topk"]),
                )
                if mode == "topology"
                else classification.new_zeros(())
            )
            anchor = prototype_anchor_loss(probe, initial_weight)
            loss = (
                classification
                + float(section["lambda_topology"]) * relation
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
            totals["loss"] += float(loss.detach()) * count
            totals["classification"] += (
                float(classification.detach()) * count
            )
            totals["topology"] += float(relation.detach()) * count
            totals["anchor"] += float(anchor.detach()) * count
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
            **{
                key + "_loss": value / seen
                for key, value in totals.items()
            },
            **metrics,
        }
        history.append(row)
        print(
            "Topology epoch %03d loss=%.4f cls=%.4f "
            "relation=%.4f anchor=%.4f val_acc=%.2f val_f1=%.2f"
            % (
                epoch + 1,
                row["loss_loss"],
                row["classification_loss"],
                row["topology_loss"],
                row["anchor_loss"],
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
        "topology_diagnostics": {
            "mean_similarity": float(topology.mean()),
            "max_similarity": float(topology.max()),
            "atomic_mean": float(atomic.mean()),
            "hyperedge_mean": float(edge.mean()),
            "neighbor_topk": int(section["neighbor_topk"]),
        },
    }
