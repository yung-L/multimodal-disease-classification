from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .common import normalize


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        scale: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )
        nn.init.zeros_(self.network[-1].weight)
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return normalize(value + self.scale * self.network(value))


class S2DPG(nn.Module):
    """MVPDR classifier with candidate-conditioned symptom verification."""

    def __init__(
        self,
        feature_dim: int,
        classnames,
        symptom_features: torch.Tensor,
        class_text_features: torch.Tensor,
        class_text_labels: torch.Tensor,
        graph: dict,
        cfg: dict,
    ):
        super().__init__()
        self.cfg = cfg
        self.ablation = cfg.get("ablation", "none")
        self.text_aggregation = cfg.get(
            "text_aggregation", "mean_max"
        )
        if self.text_aggregation not in {"mean", "max", "mean_max"}:
            raise ValueError(
                "text_aggregation must be one of: mean, max, mean_max"
            )
        self.classnames = list(classnames)
        self.num_classes = len(classnames)
        self.num_symptoms = len(graph["symptoms"])
        self.num_components = int(cfg["visual_components"])

        # Strong MVPDR base: learnable visual clusters and all official
        # class-description prototypes.
        self.visual_prototypes = nn.Parameter(
            torch.zeros(
                self.num_classes,
                self.num_components,
                feature_dim,
            )
        )
        self.register_buffer(
            "component_mask",
            torch.zeros(
                self.num_classes,
                self.num_components,
                dtype=torch.bool,
            ),
        )
        self.class_text_features = nn.Parameter(
            normalize(class_text_features.clone())
        )
        self.register_buffer(
            "class_text_labels", class_text_labels.long()
        )

        # Symptom matching uses an independent projection so its supervision
        # cannot deform the MVPDR visual classification space.
        self.symptom_adapter = ResidualAdapter(
            feature_dim,
            int(cfg["symptom_adapter_hidden"]),
            float(cfg["dropout"]),
            float(cfg["symptom_adapter_scale"]),
        )
        self.register_buffer(
            "symptom_features", normalize(symptom_features)
        )
        self.register_buffer(
            "positive_matrix",
            torch.tensor(
                graph["positive_matrix"], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "negative_matrix",
            torch.tensor(
                graph["negative_matrix"], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "symptom_idf",
            torch.tensor(
                [item["idf"] for item in graph["symptoms"]],
                dtype=torch.float32,
            ),
        )
        attribute_names = sorted({
            item["attribute_type"] for item in graph["symptoms"]
        })
        attribute_index = {
            name: index for index, name in enumerate(attribute_names)
        }
        self.register_buffer(
            "symptom_attribute_index",
            torch.tensor([
                attribute_index[item["attribute_type"]]
                for item in graph["symptoms"]
            ], dtype=torch.long),
        )
        self.attribute_log_weight = nn.Parameter(
            torch.zeros(len(attribute_names))
        )

        hyperedges = graph.get("composite_hyperedges", [])
        hyperedge_components = torch.zeros(
            len(hyperedges), self.num_symptoms, dtype=torch.float32
        )
        hyperedge_classes = torch.zeros(
            len(hyperedges), self.num_classes, dtype=torch.float32
        )
        for edge_id, edge in enumerate(hyperedges):
            hyperedge_components[
                edge_id, edge["atomic_symptom_indices"]
            ] = 1.0
            hyperedge_classes[
                edge_id, int(edge["class_id"])
            ] = 1.0
        hyperedge_sizes = hyperedge_components.sum(dim=1).clamp_min(1)
        hyperedge_distinctiveness = self._edge_distinctiveness(
            hyperedge_components,
            hyperedge_classes,
            torch.tensor(
                graph["positive_matrix"], dtype=torch.float32
            ),
        )
        keep = (
            hyperedge_distinctiveness
            >= float(cfg.get("hyperedge_min_distinctiveness", 0.20))
        )
        self.register_buffer(
            "hyperedge_components", hyperedge_components[keep]
        )
        self.register_buffer(
            "hyperedge_classes", hyperedge_classes[keep]
        )
        self.register_buffer(
            "hyperedge_sizes", hyperedge_sizes[keep]
        )
        self.register_buffer(
            "hyperedge_distinctiveness",
            hyperedge_distinctiveness[keep],
        )

        # Bounded residual strengths. Initial corrections are deliberately
        # small; validation decides whether they grow.
        self.atomic_strength_raw = nn.Parameter(torch.tensor(-1.5))
        self.hyperedge_strength_raw = nn.Parameter(torch.tensor(-1.5))
        self.register_buffer(
            "confusion_neighbors",
            torch.zeros(
                self.num_classes,
                int(cfg["confusion_neighbors"]),
                dtype=torch.long,
            ),
        )
        self.initialized = False

    @staticmethod
    def _edge_distinctiveness(
        components: torch.Tensor,
        edge_classes: torch.Tensor,
        positive_matrix: torch.Tensor,
    ) -> torch.Tensor:
        if len(components) == 0:
            return torch.zeros(0)
        target = edge_classes.argmax(dim=1)
        overlap = (
            components[:, None, :] * positive_matrix[None, :, :]
        ).sum(dim=2) / components.sum(dim=1, keepdim=True).clamp_min(1)
        overlap[
            torch.arange(len(components)), target
        ] = -1.0
        return (1.0 - overlap.max(dim=1).values).clamp(0, 1)

    @torch.no_grad()
    def initialize_visual_components(
        self,
        global_features: torch.Tensor,
        labels: torch.Tensor,
        seed: int,
    ) -> None:
        features = normalize(global_features).cpu()
        centers = torch.zeros_like(
            self.visual_prototypes, device="cpu"
        )
        masks = torch.zeros_like(
            self.component_mask, device="cpu"
        )
        class_means = []
        labels_cpu = labels.cpu()
        for class_id in range(self.num_classes):
            values = features[labels_cpu == class_id]
            if len(values) == 0:
                raise RuntimeError(
                    "No training images for class %d" % class_id
                )
            components = min(self.num_components, len(values))
            if components == 1:
                class_centers = normalize(
                    values.mean(dim=0, keepdim=True)
                )
            else:
                estimator = KMeans(
                    n_clusters=components,
                    random_state=seed + class_id,
                    n_init=10,
                )
                estimator.fit(values.numpy())
                class_centers = normalize(
                    torch.from_numpy(
                        estimator.cluster_centers_
                    ).float()
                )
            centers[class_id, :components] = class_centers
            masks[class_id, :components] = True
            if components < self.num_components:
                centers[class_id, components:] = class_centers[0]
            class_means.append(normalize(values.mean(dim=0)))
        self.visual_prototypes.copy_(
            centers.to(self.visual_prototypes.device)
        )
        self.component_mask.copy_(
            masks.to(self.component_mask.device)
        )
        class_means = torch.stack(class_means)
        similarity = class_means @ class_means.t()
        similarity.fill_diagonal_(-1e4)
        self.confusion_neighbors.copy_(
            similarity.topk(
                self.confusion_neighbors.shape[1], dim=1
            ).indices.to(self.confusion_neighbors.device)
        )
        self.initialized = True

    def _aggregate_text(
        self, each_logit: torch.Tensor, mode: str
    ) -> torch.Tensor:
        output = []
        for class_id in range(self.num_classes):
            selected = each_logit[
                :, self.class_text_labels == class_id
            ]
            if mode == "mean":
                output.append(selected.mean(dim=1))
            else:
                output.append(selected.max(dim=1).values)
        return torch.stack(output, dim=1)

    def mvpdr_logits(self, global_features: torch.Tensor):
        features = normalize(global_features)
        visual = normalize(self.visual_prototypes)
        affinity = torch.einsum("bd,ckd->bck", features, visual)
        beta = float(self.cfg["mvpdr_beta"])
        visual_each = torch.exp(-beta * (1.0 - affinity))
        visual_each = visual_each.masked_fill(
            ~self.component_mask[None, :, :], 0.0
        )
        visual_logits = visual_each.sum(dim=2)

        text_each = (
            float(self.cfg["mvpdr_text_scale"])
            * features
            @ normalize(self.class_text_features).t()
        )
        text_mean = self._aggregate_text(text_each, "mean")
        text_max = self._aggregate_text(text_each, "max")
        if self.text_aggregation == "mean":
            text_logits = text_mean
        elif self.text_aggregation == "max":
            text_logits = text_max
        else:
            text_logits = (
                float(self.cfg["mvpdr_gamma"]) * text_mean
                + float(self.cfg["mvpdr_beta"]) * text_max
            )
        baseline = (
            text_logits
            + float(self.cfg["mvpdr_alpha"]) * visual_logits
        )
        return baseline, visual_logits, text_mean, text_max

    def symptom_evidence(self, region_features: torch.Tensor):
        projected = self.symptom_adapter(region_features)
        similarity = torch.einsum(
            "brd,md->brm", projected, self.symptom_features
        )
        attention = F.softmax(
            similarity
            / float(self.cfg["region_attention_temperature"]),
            dim=1,
        )
        raw = (attention * similarity).sum(dim=1)
        standardized = (
            raw - raw.mean(dim=1, keepdim=True)
        ) / raw.std(dim=1, keepdim=True).clamp_min(1e-6)
        evidence = torch.sigmoid(
            standardized
            / float(self.cfg.get("symptom_evidence_temperature", 1.0))
        )
        return standardized, evidence

    def hyperedge_activation(
        self, evidence: torch.Tensor
    ) -> torch.Tensor:
        if self.hyperedge_components.shape[0] == 0:
            return evidence.new_zeros(evidence.shape[0], 0)
        temperature = float(
            self.cfg.get("hyperedge_softmin_temperature", 0.10)
        )
        values = evidence[:, None, :].expand(
            -1, self.hyperedge_components.shape[0], -1
        )
        values = values.masked_fill(
            ~self.hyperedge_components.bool()[None, :, :],
            float("inf"),
        )
        soft_min = -temperature * torch.logsumexp(
            -values / temperature, dim=2
        )
        soft_min = soft_min + temperature * self.hyperedge_sizes.log()[
            None, :
        ]
        return soft_min.clamp(0, 1)

    @staticmethod
    def _candidate_standardize(
        score: torch.Tensor,
    ) -> torch.Tensor:
        return (
            score - score.mean(dim=1, keepdim=True)
        ) / score.std(dim=1, keepdim=True).clamp_min(1e-6)

    def candidate_verification(
        self,
        baseline_logits: torch.Tensor,
        evidence: torch.Tensor,
        edge_activation: torch.Tensor,
    ):
        candidate_k = min(
            int(self.cfg["candidate_topk"]), self.num_classes
        )
        candidate_ids = baseline_logits.topk(
            candidate_k, dim=1
        ).indices
        membership = self.positive_matrix[candidate_ids]
        pair_difference = (
            membership[:, :, None, :]
            - membership[:, None, :, :]
        )
        attribute_weight = self.attribute_log_weight.exp().clamp(
            0.25, 4.0
        )[self.symptom_attribute_index]
        differential_weight = (
            pair_difference.abs()
            * self.symptom_idf[None, None, None, :]
            * attribute_weight[None, None, None, :]
        )
        centered_evidence = evidence - 0.5
        pair_atomic = (
            pair_difference
            * differential_weight
            * centered_evidence[:, None, None, :]
        ).sum(dim=3) / differential_weight.sum(dim=3).clamp_min(1e-6)
        opponent_mask = (
            ~torch.eye(
                candidate_k,
                dtype=torch.bool,
                device=baseline_logits.device,
            )
        )[None, :, :]
        atomic_score = (
            pair_atomic.masked_fill(~opponent_mask, 0.0).sum(dim=2)
            / max(candidate_k - 1, 1)
        )

        edge_score_all = baseline_logits.new_zeros(
            baseline_logits.shape
        )
        edge_denominator = baseline_logits.new_zeros(
            self.num_classes
        )
        if edge_activation.shape[1] > 0:
            weighted_edge = (
                edge_activation
                * self.hyperedge_distinctiveness[None, :]
            )
            edge_score_all = weighted_edge @ self.hyperedge_classes
            edge_denominator = (
                self.hyperedge_distinctiveness[:, None]
                * self.hyperedge_classes
            ).sum(dim=0)
            edge_score_all = edge_score_all / edge_denominator[
                None, :
            ].clamp_min(1e-6)
        candidate_edge = edge_score_all.gather(1, candidate_ids)
        pair_edge = (
            candidate_edge[:, :, None]
            - candidate_edge[:, None, :]
        )
        edge_score = (
            pair_edge.masked_fill(~opponent_mask, 0.0).sum(dim=2)
            / max(candidate_k - 1, 1)
        )

        atomic_score = self._candidate_standardize(atomic_score)
        edge_score = self._candidate_standardize(edge_score)
        verifier_logits = (
            float(self.cfg["verifier_atomic_weight"]) * atomic_score
            + float(self.cfg["verifier_hyperedge_weight"]) * edge_score
        )
        candidate_correction = baseline_logits.new_zeros(
            baseline_logits.shape
        )
        atomic_strength = (
            float(self.cfg["max_atomic_correction"])
            * self.atomic_strength_raw.sigmoid()
        )
        edge_strength = (
            float(self.cfg["max_hyperedge_correction"])
            * self.hyperedge_strength_raw.sigmoid()
        )
        if self.ablation in {
            "mvpdr_only",
            "no_symptom_branch",
            "no_atomic_rerank",
        }:
            atomic_strength = atomic_strength * 0.0
        if self.ablation in {
            "mvpdr_only",
            "no_symptom_branch",
            "no_hyperedge",
        }:
            edge_strength = edge_strength * 0.0
        candidate_value = (
            atomic_strength
            * float(self.cfg["verifier_atomic_weight"])
            * atomic_score
            + edge_strength
            * float(self.cfg["verifier_hyperedge_weight"])
            * edge_score
        )
        candidate_correction.scatter_(
            1, candidate_ids, candidate_value
        )
        return (
            baseline_logits + candidate_correction,
            candidate_ids,
            atomic_score,
            edge_score,
            verifier_logits,
            candidate_correction,
            atomic_strength,
            edge_strength,
        )

    def forward(
        self, region_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if not self.initialized:
            raise RuntimeError(
                "Initialize visual components before training"
            )
        global_features = normalize(region_features[:, 0])
        (
            baseline_logits,
            visual_logits,
            text_mean,
            text_max,
        ) = self.mvpdr_logits(global_features)
        symptom_logits, evidence = self.symptom_evidence(
            region_features
        )
        edge_activation = self.hyperedge_activation(evidence)
        (
            logits,
            candidate_ids,
            atomic_score,
            edge_score,
            verifier_logits,
            correction,
            atomic_strength,
            edge_strength,
        ) = self.candidate_verification(
            baseline_logits, evidence, edge_activation
        )
        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "visual_logits": visual_logits,
            "text_mean_logits": text_mean,
            "text_max_logits": text_max,
            "symptom_logits": symptom_logits,
            "symptom_evidence": evidence,
            "hyperedge_activation": edge_activation,
            "candidate_ids": candidate_ids,
            "atomic_candidate_score": atomic_score,
            "hyperedge_candidate_score": edge_score,
            "verifier_candidate_logits": verifier_logits,
            "candidate_correction": correction,
            "atomic_strength": atomic_strength,
            "hyperedge_strength": edge_strength,
        }

    def mvpdr_auxiliary_loss(
        self, output: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        visual_loss = (
            float(self.cfg["loss_visual"])
            * F.cross_entropy(output["visual_logits"], labels)
        )
        mean_weight = float(self.cfg["loss_text_mean"])
        max_weight = float(self.cfg["loss_text_max"])
        # Keep the total text-auxiliary weight constant across variants so
        # that the ablation changes only the aggregation rule.
        if self.text_aggregation == "mean":
            text_loss = (mean_weight + max_weight) * F.cross_entropy(
                output["text_mean_logits"], labels
            )
        elif self.text_aggregation == "max":
            text_loss = (mean_weight + max_weight) * F.cross_entropy(
                output["text_max_logits"], labels
            )
        else:
            text_loss = (
                mean_weight
                * F.cross_entropy(output["text_mean_logits"], labels)
                + max_weight
                * F.cross_entropy(output["text_max_logits"], labels)
            )
        return visual_loss + text_loss

    def symptom_rank_loss(
        self, symptom_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.ablation in {"mvpdr_only", "no_symptom_branch"}:
            return symptom_logits.new_zeros(())
        positive = self.positive_matrix[labels].bool()
        losses = []
        margin = float(self.cfg["symptom_rank_margin"])
        for index in range(len(labels)):
            if not positive[index].any():
                continue
            neighbors = self.confusion_neighbors[labels[index]]
            competitor = (
                self.positive_matrix[neighbors].sum(dim=0).bool()
                & ~positive[index]
            )
            if not competitor.any():
                continue
            positive_score = symptom_logits[index][
                positive[index]
            ].topk(
                min(3, int(positive[index].sum()))
            ).values.mean()
            negative_score = symptom_logits[index][competitor].max()
            losses.append(
                F.softplus(
                    margin - positive_score + negative_score
                )
            )
        if not losses:
            return symptom_logits.new_zeros(())
        return torch.stack(losses).mean()

    def hyperedge_rank_loss(
        self, activation: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if (
            activation.shape[1] == 0
            or self.ablation in {
                "mvpdr_only",
                "no_symptom_branch",
                "no_hyperedge",
            }
        ):
            return activation.new_zeros(())
        edge_class = self.hyperedge_classes.argmax(dim=1)
        losses = []
        margin = float(self.cfg["hyperedge_rank_margin"])
        for index in range(len(labels)):
            positive = edge_class == labels[index]
            if not positive.any():
                continue
            neighbors = self.confusion_neighbors[labels[index]]
            negative = (
                edge_class[:, None] == neighbors[None, :]
            ).any(dim=1)
            if not negative.any():
                continue
            positive_score = (
                activation[index][positive]
                * self.hyperedge_distinctiveness[positive]
            ).max()
            negative_score = (
                activation[index][negative]
                * self.hyperedge_distinctiveness[negative]
            ).max()
            losses.append(
                F.softplus(
                    margin - positive_score + negative_score
                )
            )
        if not losses:
            return activation.new_zeros(())
        return torch.stack(losses).mean()

    def correction_margin_loss(
        self,
        output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.ablation in {"mvpdr_only", "no_symptom_branch"}:
            return output["logits"].new_zeros(())
        candidates = output["candidate_ids"]
        true_position = candidates.eq(labels[:, None])
        valid = true_position.any(dim=1)
        if not valid.any():
            return output["logits"].new_zeros(())
        target_position = true_position.float().argmax(dim=1)
        return F.cross_entropy(
            output["verifier_candidate_logits"][valid],
            target_position[valid],
        )

    def diversity_loss(self) -> torch.Tensor:
        components = normalize(self.visual_prototypes)
        losses = []
        for class_id in range(self.num_classes):
            valid = components[class_id][
                self.component_mask[class_id]
            ]
            if len(valid) < 2:
                continue
            similarity = valid @ valid.t()
            mask = ~torch.eye(
                len(valid),
                dtype=torch.bool,
                device=valid.device,
            )
            losses.append(
                F.relu(similarity[mask] - 0.80).mean()
            )
        if not losses:
            return components.new_zeros(())
        return torch.stack(losses).mean()
