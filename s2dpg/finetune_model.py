from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import normalize


class SymptomGuidedCLIP(nn.Module):
    """Partially fine-tuned CLIP with structured symptom supervision."""

    def __init__(
        self,
        clip_model,
        class_text_features: torch.Tensor,
        class_text_labels: torch.Tensor,
        symptom_features: torch.Tensor,
        graph: dict,
        cfg: dict,
        mode: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.student = clip_model.float()
        self.teacher = copy.deepcopy(clip_model).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        if not hasattr(self.student.visual, "layer4"):
            raise RuntimeError(
                "Partial fine-tuning currently requires a CLIP ResNet backbone"
            )
        self.tune_layer4 = bool(cfg.get("tune_layer4", True))
        self.tune_attnpool = bool(cfg.get("tune_attnpool", True))
        if self.tune_layer4:
            for parameter in self.student.visual.layer4.parameters():
                parameter.requires_grad_(True)
        if self.tune_attnpool:
            for parameter in self.student.visual.attnpool.parameters():
                parameter.requires_grad_(True)

        self.class_prototypes = nn.Parameter(
            normalize(class_text_features.clone())
        )
        self.register_buffer(
            "class_text_labels", class_text_labels.long()
        )
        self.register_buffer(
            "initial_class_prototypes",
            normalize(class_text_features.clone()),
        )
        self.register_buffer(
            "symptom_features", normalize(symptom_features)
        )
        self.register_buffer(
            "positive_matrix",
            torch.tensor(graph["positive_matrix"], dtype=torch.float32),
        )
        self.num_classes = len(graph["classnames"])
        self.num_symptoms = len(graph["symptoms"])
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(float(cfg["init_logit_scale"])))
        )
        dim = symptom_features.shape[1]
        self.symptom_projection = nn.Sequential(
            nn.Linear(dim, int(cfg["symptom_hidden"]), bias=False),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(int(cfg["symptom_hidden"]), dim, bias=False),
        )
        nn.init.zeros_(self.symptom_projection[-1].weight)
        self.symptom_projection_scale = nn.Parameter(
            torch.tensor(float(cfg["symptom_projection_scale"]))
        )
        self.symptom_logit_scale = nn.Parameter(
            torch.tensor(float(cfg.get("symptom_logit_scale", 10.0))).log()
        )
        self.symptom_logit_bias = nn.Parameter(
            torch.tensor(float(cfg.get("symptom_logit_bias", 0.0)))
        )
        self.latent_strength_raw = nn.Parameter(torch.tensor(-4.0))
        self.hierarchy_atomic_strength_raw = nn.Parameter(torch.tensor(-2.0))
        self.hierarchy_hyperedge_strength_raw = nn.Parameter(
            torch.tensor(-2.0)
        )
        symptom_frequency = self.positive_matrix.sum(dim=0)
        self.register_buffer(
            "symptom_idf",
            torch.log(
                (1.0 + self.num_classes)
                / (1.0 + symptom_frequency)
            ) + 1.0,
        )

        edges = graph.get("composite_hyperedges", [])
        edge_components = torch.zeros(
            len(edges), self.num_symptoms, dtype=torch.float32
        )
        edge_classes = torch.zeros(
            len(edges), self.num_classes, dtype=torch.float32
        )
        for edge_id, edge in enumerate(edges):
            edge_components[
                edge_id, edge["atomic_symptom_indices"]
            ] = 1.0
            edge_classes[edge_id, int(edge["class_id"])] = 1.0
        edge_sizes = edge_components.sum(dim=1).clamp_min(1)
        distinctiveness = self._edge_distinctiveness(
            edge_components, edge_classes, self.positive_matrix
        )
        keep = distinctiveness >= float(
            cfg["hyperedge_min_distinctiveness"]
        )
        self.register_buffer(
            "edge_components", edge_components[keep]
        )
        self.register_buffer("edge_classes", edge_classes[keep])
        self.register_buffer("edge_sizes", edge_sizes[keep])
        self.register_buffer(
            "edge_distinctiveness", distinctiveness[keep]
        )
        if keep.any():
            edge_embeddings = normalize(
                edge_components[keep] @ self.symptom_features
                / edge_sizes[keep, None]
            )
        else:
            edge_embeddings = self.symptom_features.new_zeros(
                (0, self.symptom_features.shape[1])
            )
        self.register_buffer("edge_embeddings", edge_embeddings)
        atomic_graph, full_graph = self._build_class_graphs(
            self.positive_matrix,
            graph.get("composite_hyperedges", []),
            self.symptom_idf,
        )
        self.register_buffer(
            "atomic_class_graph", atomic_graph
        )
        self.register_buffer(
            "full_class_graph", full_graph
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.student.eval()
        if mode and not (
            self.mode in {"latent", "latent_no_hyperedge"}
            and self.cfg.get("latent_freeze_base", True)
        ):
            if self.tune_layer4:
                self.student.visual.layer4.train()
            if self.tune_attnpool:
                self.student.visual.attnpool.train()
        if mode:
            self.symptom_projection.train()
        return self

    @staticmethod
    def _edge_distinctiveness(components, classes, positive):
        if len(components) == 0:
            return torch.zeros(0)
        target = classes.argmax(dim=1)
        overlap = (
            components[:, None, :] * positive[None, :, :]
        ).sum(dim=2) / components.sum(dim=1, keepdim=True).clamp_min(1)
        overlap[torch.arange(len(components)), target] = -1
        return (1 - overlap.max(dim=1).values).clamp(0, 1)

    @staticmethod
    def _build_class_graphs(
        positive: torch.Tensor,
        hyperedges: list,
        symptom_idf: torch.Tensor,
    ):
        weighted = positive * symptom_idf[None, :]
        intersection = weighted @ positive.t()
        mass = weighted.sum(dim=1)
        union = (
            mass[:, None]
            + mass[None, :]
            - intersection
        ).clamp_min(1e-6)
        atomic = intersection / union
        atomic.fill_diagonal_(0.0)

        num_classes = positive.shape[0]
        edge_sets = [[] for _ in range(num_classes)]
        for edge in hyperedges:
            signature = set(edge["atomic_symptom_indices"])
            if len(signature) >= 2:
                edge_sets[int(edge["class_id"])].append(signature)
        hyper = torch.zeros(
            num_classes, num_classes, dtype=torch.float32
        )
        for first in range(num_classes):
            for second in range(first + 1, num_classes):
                if not edge_sets[first] or not edge_sets[second]:
                    continue
                similarities = []
                for first_edge in edge_sets[first]:
                    for second_edge in edge_sets[second]:
                        union_set = first_edge | second_edge
                        similarities.append(
                            len(first_edge & second_edge)
                            / max(len(union_set), 1)
                        )
                similarity = max(similarities)
                hyper[first, second] = similarity
                hyper[second, first] = similarity
        full = (
            atomic
            + 0.25 * hyper
        ).clamp(0, 1)
        full.fill_diagonal_(0.0)
        return atomic, full

    def trainable_groups(self):
        if (
            self.mode in {"latent", "latent_no_hyperedge"}
            and self.cfg.get("latent_freeze_base", True)
        ):
            for parameter in self.student.visual.layer4.parameters():
                parameter.requires_grad_(False)
            for parameter in self.student.visual.attnpool.parameters():
                parameter.requires_grad_(False)
            self.class_prototypes.requires_grad_(False)
            self.logit_scale.requires_grad_(False)
            head = list(self.symptom_projection.parameters())
            head += [
                self.symptom_projection_scale,
                self.symptom_logit_scale,
                self.symptom_logit_bias,
                self.latent_strength_raw,
            ]
            return [], head
        backbone = []
        if self.tune_layer4:
            backbone += list(self.student.visual.layer4.parameters())
        if self.tune_attnpool:
            backbone += list(self.student.visual.attnpool.parameters())
        head = [self.class_prototypes, self.logit_scale]
        head += list(self.symptom_projection.parameters())
        head += [
            self.symptom_projection_scale,
            self.symptom_logit_scale,
            self.symptom_logit_bias,
            self.latent_strength_raw,
            self.hierarchy_atomic_strength_raw,
            self.hierarchy_hyperedge_strength_raw,
        ]
        return backbone, head

    def encode_student(self, images: torch.Tensor):
        batch, regions, channels, height, width = images.shape
        flattened = images.reshape(
            batch * regions, channels, height, width
        )
        encoded = normalize(self.student.encode_image(flattened))
        return encoded.reshape(batch, regions, -1)

    @torch.no_grad()
    def encode_teacher(self, global_images: torch.Tensor):
        return normalize(self.teacher.encode_image(global_images))

    def symptom_scores(self, regions: torch.Tensor):
        residual = self.symptom_projection(regions)
        regions = normalize(
            regions + self.symptom_projection_scale * residual
        )
        similarity = torch.einsum(
            "brd,md->brm", regions, self.symptom_features
        )
        attention = F.softmax(
            similarity
            / float(self.cfg["region_attention_temperature"]),
            dim=1,
        )
        image_score = (attention * similarity).sum(dim=1)
        scale = self.symptom_logit_scale.exp().clamp(1, 100)
        return (
            scale * image_score + self.symptom_logit_bias,
            scale * similarity + self.symptom_logit_bias,
        )

    def aggregate_class_descriptions(
        self, description_logits: torch.Tensor
    ):
        means, maxima = [], []
        for class_id in range(self.num_classes):
            selected = description_logits[
                :, self.class_text_labels == class_id
            ]
            means.append(selected.mean(dim=1))
            maxima.append(selected.max(dim=1).values)
        return torch.stack(means, dim=1), torch.stack(maxima, dim=1)

    def class_mean_prototypes(self):
        prototypes = []
        normalized = normalize(self.class_prototypes)
        for class_id in range(self.num_classes):
            prototypes.append(normalize(
                normalized[self.class_text_labels == class_id].mean(dim=0)
            ))
        return torch.stack(prototypes)

    def hyperedge_scores(self, symptom_scores: torch.Tensor):
        if self.edge_components.shape[0] == 0:
            return symptom_scores.new_zeros(
                symptom_scores.shape[0], 0
            )
        probability = torch.sigmoid(
            symptom_scores
            / float(self.cfg["symptom_temperature"])
        )
        values = probability[:, None, :].expand(
            -1, self.edge_components.shape[0], -1
        )
        values = values.masked_fill(
            ~self.edge_components.bool()[None, :, :],
            float("inf"),
        )
        temperature = float(self.cfg["hyperedge_softmin_temperature"])
        score = -temperature * torch.logsumexp(
            -values / temperature, dim=2
        )
        score += temperature * self.edge_sizes.log()[None, :]
        return score.clamp(0, 1)

    def teacher_symptom_visibility(
        self, teacher_feature: torch.Tensor
    ):
        """Return detached frozen-teacher evidence for every symptom."""
        gate = str(
            self.cfg.get("hierarchy_visibility_gate", "soft")
        ).lower()
        if (
            not bool(self.cfg.get("hierarchy_use_visibility", True))
            or gate == "none"
        ):
            return teacher_feature.new_ones(
                teacher_feature.shape[0], self.num_symptoms
            )
        similarity = teacher_feature @ self.symptom_features.t()
        threshold = float(
            self.cfg.get("hierarchy_visibility_threshold", 0.20)
        )
        temperature = float(
            self.cfg.get("hierarchy_visibility_temperature", 0.07)
        )
        soft_visibility = torch.sigmoid(
            (similarity - threshold) / max(temperature, 1e-6)
        )
        if gate == "soft":
            return soft_visibility.detach()
        if gate == "hard":
            floor = float(
                self.cfg.get("hierarchy_visibility_floor", 0.50)
            )
            return (soft_visibility >= floor).to(
                soft_visibility.dtype
            ).detach()
        raise ValueError(
            "hierarchy_visibility_gate must be none, hard or soft; got %r"
            % gate
        )

    def hierarchical_dynamic_logits(
        self,
        global_feature: torch.Tensor,
        base_logits: torch.Tensor,
        visibility: torch.Tensor,
    ):
        """Build global -> atomic -> composite image-conditioned logits."""
        batch = global_feature.shape[0]
        membership = self.positive_matrix[None, :, :].expand(
            batch, -1, -1
        )
        evidence = visibility[:, None, :].expand(
            -1, self.num_classes, -1
        )
        weights = (
            membership
            * evidence
            * self.symptom_idf[None, None, :]
        )

        symptom_topk = min(
            int(self.cfg.get("hierarchy_symptom_topk", 4)),
            self.num_symptoms,
        )
        selected_ids = weights.masked_fill(
            membership <= 0, -1e4
        ).topk(symptom_topk, dim=2).indices
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(
            2, selected_ids, weights.gather(2, selected_ids)
        )
        sparse_weights = sparse_weights * membership
        atomic_mass = sparse_weights.sum(dim=2, keepdim=True)
        atomic_prototype = normalize(
            torch.einsum(
                "bcm,md->bcd",
                sparse_weights,
                self.symptom_features,
            )
            / atomic_mass.clamp_min(1e-6)
        )
        selected_visibility = torch.zeros_like(weights)
        selected_visibility.scatter_(
            2,
            selected_ids,
            evidence.gather(2, selected_ids),
        )
        selected_visibility = selected_visibility * membership
        selected_count = (sparse_weights > 0).sum(
            dim=2
        ).clamp_min(1)
        atomic_confidence = (
            selected_visibility.sum(dim=2) / selected_count
        )
        atomic_confidence = atomic_confidence * (
            atomic_mass.squeeze(2) > 0
        )

        class_base = self.class_mean_prototypes()
        atomic_class = normalize(
            class_base[None, :, :]
            + float(self.cfg.get("hierarchy_atomic_scale", 0.25))
            * atomic_prototype
        )
        scale = self.logit_scale.exp().clamp(1, 100)
        class_base_score = scale * torch.einsum(
            "bd,cd->bc", global_feature, class_base
        )
        atomic_score = scale * torch.einsum(
            "bd,bcd->bc", global_feature, atomic_class
        )
        delta_temperature = float(
            self.cfg.get("hierarchy_delta_temperature", 1.0)
        )
        atomic_delta = torch.tanh(
            (atomic_score - class_base_score)
            / max(delta_temperature, 1e-6)
        )
        atomic_strength = (
            float(
                self.cfg.get(
                    "hierarchy_max_atomic_correction", 1.0
                )
            )
            * self.hierarchy_atomic_strength_raw.sigmoid()
        )
        atomic_correction = (
            atomic_strength * atomic_confidence * atomic_delta
        )
        atomic_logits = base_logits + atomic_correction

        composite_logits = atomic_logits
        hyperedge_correction = torch.zeros_like(base_logits)
        hyperedge_confidence = torch.zeros_like(base_logits)
        hyperedge_strength = (
            float(
                self.cfg.get(
                    "hierarchy_max_hyperedge_correction", 1.0
                )
            )
            * self.hierarchy_hyperedge_strength_raw.sigmoid()
        )
        use_hyperedges = bool(
            self.cfg.get("hierarchy_use_hyperedges", True)
        )
        if use_hyperedges and self.edge_components.shape[0] > 0:
            component_mask = self.edge_components.bool()[None, :, :]
            expanded_visibility = visibility[:, None, :].expand(
                -1, self.edge_components.shape[0], -1
            )
            edge_visibility = expanded_visibility.masked_fill(
                ~component_mask, 1.0
            ).min(dim=2).values
            edge_weight = (
                edge_visibility
                * self.edge_distinctiveness[None, :]
            )
            edge_mass = torch.einsum(
                "bh,hc->bc", edge_weight, self.edge_classes
            )
            hyperedge_prototype = normalize(
                torch.einsum(
                    "bh,hc,hd->bcd",
                    edge_weight,
                    self.edge_classes,
                    self.edge_embeddings,
                )
                / edge_mass[:, :, None].clamp_min(1e-6)
            )
            composite_class = normalize(
                class_base[None, :, :]
                + float(
                    self.cfg.get("hierarchy_atomic_scale", 0.25)
                )
                * atomic_prototype
                + float(
                    self.cfg.get("hierarchy_hyperedge_scale", 0.15)
                )
                * hyperedge_prototype
            )
            composite_score = scale * torch.einsum(
                "bd,bcd->bc", global_feature, composite_class
            )
            hyperedge_delta = torch.tanh(
                (composite_score - atomic_score)
                / max(delta_temperature, 1e-6)
            )
            hyperedge_confidence = edge_mass.clamp(0, 1)
            hyperedge_correction = (
                hyperedge_strength
                * hyperedge_confidence
                * hyperedge_delta
            )
            composite_logits = atomic_logits + hyperedge_correction

        hierarchy_logits = torch.stack(
            [base_logits, atomic_logits, composite_logits], dim=1
        )
        transition_weights = torch.stack(
            [atomic_confidence, hyperedge_confidence], dim=1
        )
        return {
            "logits": composite_logits,
            "hierarchy_logits": hierarchy_logits,
            "hierarchy_transition_weights": transition_weights,
            "hierarchy_atomic_correction": atomic_correction,
            "hierarchy_hyperedge_correction": hyperedge_correction,
            "hierarchy_atomic_strength": atomic_strength,
            "hierarchy_hyperedge_strength": hyperedge_strength,
        }

    def latent_dynamic_logits(
        self,
        global_feature: torch.Tensor,
        base_logits: torch.Tensor,
        symptom_scores: torch.Tensor,
        edge_scores: torch.Tensor,
    ):
        batch = len(global_feature)
        candidate_k = min(
            int(self.cfg["latent_candidate_topk"]),
            self.num_classes,
        )
        candidate_ids = base_logits.topk(
            candidate_k, dim=1
        ).indices
        membership = self.positive_matrix[candidate_ids]
        standardized = (
            symptom_scores - symptom_scores.mean(dim=1, keepdim=True)
        ) / symptom_scores.std(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        evidence = torch.sigmoid(
            standardized
            / float(self.cfg["latent_evidence_temperature"])
        )
        candidate_frequency = membership.mean(dim=1, keepdim=True)
        distinctiveness = (
            1.0 - candidate_frequency
        ).clamp_min(float(self.cfg["latent_distinctiveness_floor"]))
        weights = (
            membership
            * evidence[:, None, :]
            * distinctiveness
            * self.symptom_idf[None, None, :]
        )

        if (
            self.mode != "latent_no_hyperedge"
            and edge_scores.shape[1] > 0
        ):
            edge_atomic_bonus = torch.einsum(
                "bh,hc,hm->bcm",
                edge_scores * self.edge_distinctiveness[None, :],
                self.edge_classes,
                self.edge_components,
            )
            batch_index = torch.arange(
                batch, device=global_feature.device
            )[:, None]
            candidate_bonus = edge_atomic_bonus[
                batch_index, candidate_ids
            ]
            weights = weights * (
                1.0
                + float(self.cfg["latent_hyperedge_boost"])
                * candidate_bonus
            )

        select_k = min(
            int(self.cfg["latent_symptom_topk"]),
            self.num_symptoms,
        )
        masked = weights.masked_fill(membership <= 0, -1e4)
        selected = masked.topk(select_k, dim=2).indices
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(
            2, selected, weights.gather(2, selected)
        )
        sparse_weights = sparse_weights * membership
        denominator = sparse_weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        symptom_prototype = normalize(
            torch.einsum(
                "bkm,md->bkd",
                sparse_weights,
                self.symptom_features,
            ) / denominator
        )
        class_prototypes = self.class_mean_prototypes()
        candidate_base = class_prototypes[candidate_ids]
        dynamic_prototype = normalize(
            candidate_base
            + float(self.cfg["latent_prototype_scale"])
            * symptom_prototype
        )
        scale = self.logit_scale.exp().clamp(1, 100)
        dynamic_score = scale * torch.einsum(
            "bd,bkd->bk", global_feature, dynamic_prototype
        )
        original_score = scale * torch.einsum(
            "bd,bkd->bk", global_feature, candidate_base
        )
        prototype_delta = torch.tanh(
            (dynamic_score - original_score)
            / float(self.cfg["latent_delta_temperature"])
        )
        strength = (
            float(self.cfg["latent_max_correction"])
            * self.latent_strength_raw.sigmoid()
        )
        correction = base_logits.new_zeros(base_logits.shape)
        correction.scatter_(
            1, candidate_ids, strength * prototype_delta
        )
        return (
            base_logits + correction,
            candidate_ids,
            sparse_weights,
            correction,
            strength,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        regions = self.encode_student(images)
        global_feature = regions[:, 0]
        if (
            self.mode in {"latent", "latent_no_hyperedge"}
            and self.cfg.get("latent_freeze_base", True)
        ):
            teacher_feature = global_feature.detach()
        else:
            with torch.no_grad():
                teacher_feature = self.encode_teacher(images[:, 0])
        scale = self.logit_scale.exp().clamp(1, 100)
        description_logits = scale * global_feature @ normalize(
            self.class_prototypes
        ).t()
        text_mean, text_max = self.aggregate_class_descriptions(
            description_logits
        )
        base_logits = (
            float(self.cfg["description_mean_weight"]) * text_mean
            + float(self.cfg["description_max_weight"]) * text_max
        )
        symptoms, region_symptoms = self.symptom_scores(regions)
        edges = self.hyperedge_scores(symptoms)
        logits = base_logits
        candidate_ids = base_logits.topk(
            min(
                int(self.cfg.get("latent_candidate_topk", 10)),
                self.num_classes,
            ),
            dim=1,
        ).indices
        latent_weights = symptoms.new_zeros(
            len(symptoms), candidate_ids.shape[1], self.num_symptoms
        )
        latent_correction = torch.zeros_like(base_logits)
        latent_strength = self.latent_strength_raw.new_zeros(())
        hierarchy_logits = torch.stack(
            [base_logits, base_logits, base_logits], dim=1
        )
        hierarchy_transition_weights = base_logits.new_zeros(
            len(base_logits), 2, self.num_classes
        )
        hierarchy_atomic_correction = torch.zeros_like(base_logits)
        hierarchy_hyperedge_correction = torch.zeros_like(base_logits)
        hierarchy_atomic_strength = base_logits.new_zeros(())
        hierarchy_hyperedge_strength = base_logits.new_zeros(())
        if self.mode in {"latent", "latent_no_hyperedge"}:
            (
                logits,
                candidate_ids,
                latent_weights,
                latent_correction,
                latent_strength,
            ) = self.latent_dynamic_logits(
                global_feature,
                base_logits,
                symptoms,
                edges,
            )
        elif self.mode == "hier_mono":
            hierarchy = self.hierarchical_dynamic_logits(
                global_feature,
                base_logits,
                self.teacher_symptom_visibility(teacher_feature),
            )
            logits = hierarchy["logits"]
            hierarchy_logits = hierarchy["hierarchy_logits"]
            hierarchy_transition_weights = hierarchy[
                "hierarchy_transition_weights"
            ]
            hierarchy_atomic_correction = hierarchy[
                "hierarchy_atomic_correction"
            ]
            hierarchy_hyperedge_correction = hierarchy[
                "hierarchy_hyperedge_correction"
            ]
            hierarchy_atomic_strength = hierarchy[
                "hierarchy_atomic_strength"
            ]
            hierarchy_hyperedge_strength = hierarchy[
                "hierarchy_hyperedge_strength"
            ]
        return {
            "logits": logits,
            "base_logits": base_logits,
            "text_mean_logits": text_mean,
            "text_max_logits": text_max,
            "global_feature": global_feature,
            "teacher_feature": teacher_feature,
            "symptom_scores": symptoms,
            "region_symptom_scores": region_symptoms,
            "hyperedge_scores": edges,
            "latent_candidate_ids": candidate_ids,
            "latent_weights": latent_weights,
            "latent_correction": latent_correction,
            "latent_strength": latent_strength,
            "hierarchy_logits": hierarchy_logits,
            "hierarchy_transition_weights":
                hierarchy_transition_weights,
            "hierarchy_atomic_correction":
                hierarchy_atomic_correction,
            "hierarchy_hyperedge_correction":
                hierarchy_hyperedge_correction,
            "hierarchy_atomic_strength": hierarchy_atomic_strength,
            "hierarchy_hyperedge_strength":
                hierarchy_hyperedge_strength,
        }

    def hierarchy_losses(
        self,
        output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ):
        """Return level-alignment loss, monotonic loss and diagnostics."""
        zero = output["logits"].new_zeros(())
        if self.mode != "hier_mono":
            return zero, zero, {
                "monotonic_accuracy": zero,
                "margin_gain": zero,
                "visibility_coverage": zero,
                "atomic_monotonic_accuracy": zero,
                "atomic_margin_gain": zero,
                "atomic_visibility_coverage": zero,
                "composite_monotonic_accuracy": zero,
                "composite_margin_gain": zero,
                "composite_visibility_coverage": zero,
            }

        level_logits = output["hierarchy_logits"]
        auxiliary = 0.5 * (
            F.cross_entropy(level_logits[:, 0], labels)
            + F.cross_entropy(level_logits[:, 1], labels)
        )
        batch = labels.shape[0]
        levels = level_logits.shape[1]
        correct = level_logits.gather(
            2,
            labels[:, None, None].expand(-1, levels, 1),
        ).squeeze(2)

        if bool(self.cfg.get("hierarchy_use_hard_neighbors", True)):
            graph_rows = self.full_class_graph[labels].clone()
            graph_rows.scatter_(1, labels[:, None], -1.0)
            neighbor_k = min(
                int(self.cfg.get("hierarchy_hard_neighbors", 5)),
                self.num_classes - 1,
            )
            neighbors = graph_rows.topk(neighbor_k, dim=1).indices
            neighbor_logits = level_logits.gather(
                2,
                neighbors[:, None, :].expand(-1, levels, -1),
            )
            competitor = torch.logsumexp(
                neighbor_logits, dim=2
            )
        else:
            negative_mask = torch.ones(
                batch,
                self.num_classes,
                dtype=torch.bool,
                device=labels.device,
            )
            negative_mask.scatter_(1, labels[:, None], False)
            competitor = torch.logsumexp(
                level_logits.masked_fill(
                    ~negative_mask[:, None, :], -1e4
                ),
                dim=2,
            )

        margins = correct - competitor
        gains = margins[:, 1:] - margins[:, :-1]
        weights = output["hierarchy_transition_weights"].gather(
            2,
            labels[:, None, None].expand(-1, levels - 1, 1),
        ).squeeze(2)
        visibility_floor = float(
            self.cfg.get("hierarchy_visibility_floor", 0.50)
        )
        if bool(self.cfg.get("hierarchy_use_visibility", True)):
            weights = torch.where(
                weights >= visibility_floor,
                weights,
                torch.zeros_like(weights),
            )
        else:
            weights = (weights > 0).float()

        denominator = weights.sum()
        if denominator.item() <= 0:
            return auxiliary, zero, {
                "monotonic_accuracy": zero,
                "margin_gain": zero,
                "visibility_coverage": zero,
                "atomic_monotonic_accuracy": zero,
                "atomic_margin_gain": zero,
                "atomic_visibility_coverage": zero,
                "composite_monotonic_accuracy": zero,
                "composite_margin_gain": zero,
                "composite_visibility_coverage": zero,
            }
        target_gain = float(
            self.cfg.get("hierarchy_monotonic_margin", 0.05)
        )
        monotonic = (
            weights * F.softplus(target_gain - gains)
        ).sum() / denominator
        monotonic_accuracy = (
            weights * (gains >= 0).float()
        ).sum() / denominator
        margin_gain = (weights * gains).sum() / denominator
        coverage = (weights > 0).float().mean()

        def transition_diagnostics(index: int):
            transition_weights = weights[:, index]
            transition_denominator = transition_weights.sum()
            transition_coverage = (
                transition_weights > 0
            ).float().mean()
            if transition_denominator.item() <= 0:
                return zero, zero, transition_coverage
            transition_accuracy = (
                transition_weights
                * (gains[:, index] >= 0).float()
            ).sum() / transition_denominator
            transition_gain = (
                transition_weights * gains[:, index]
            ).sum() / transition_denominator
            return (
                transition_accuracy,
                transition_gain,
                transition_coverage,
            )

        atomic_accuracy, atomic_gain, atomic_coverage = (
            transition_diagnostics(0)
        )
        composite_accuracy, composite_gain, composite_coverage = (
            transition_diagnostics(1)
        )
        return auxiliary, monotonic, {
            "monotonic_accuracy": monotonic_accuracy,
            "margin_gain": margin_gain,
            "visibility_coverage": coverage,
            "atomic_monotonic_accuracy": atomic_accuracy,
            "atomic_margin_gain": atomic_gain,
            "atomic_visibility_coverage": atomic_coverage,
            "composite_monotonic_accuracy": composite_accuracy,
            "composite_margin_gain": composite_gain,
            "composite_visibility_coverage": composite_coverage,
        }

    def atomic_loss(
        self,
        symptom_scores: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ):
        if self.mode in {
            "class_only",
            "latent",
            "latent_no_hyperedge",
            "hier_mono",
            "graph_margin_atomic",
            "graph_margin_full",
        }:
            return symptom_scores.new_zeros(())
        competitor_logits = logits.detach().clone()
        competitor_logits.scatter_(1, labels[:, None], -1e4)
        competitors = competitor_logits.argmax(dim=1)
        positive = self.positive_matrix[labels].bool()
        competitor_only = (
            self.positive_matrix[competitors].bool() & ~positive
        )
        losses = []
        margin = float(self.cfg["atomic_margin"])
        for index in range(len(labels)):
            if not positive[index].any() or not competitor_only[index].any():
                continue
            positive_score = symptom_scores[index][positive[index]].topk(
                min(
                    int(self.cfg["atomic_positive_topk"]),
                    int(positive[index].sum()),
                )
            ).values.mean()
            negative_score = symptom_scores[index][
                competitor_only[index]
            ].max()
            losses.append(
                F.softplus(margin - positive_score + negative_score)
            )
        if not losses:
            return symptom_scores.new_zeros(())
        return torch.stack(losses).mean()

    def hyperedge_loss(
        self,
        edge_scores: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ):
        if self.mode != "full" or edge_scores.shape[1] == 0:
            return edge_scores.new_zeros(())
        edge_class = self.edge_classes.argmax(dim=1)
        competitor_logits = logits.detach().clone()
        competitor_logits.scatter_(1, labels[:, None], -1e4)
        competitors = competitor_logits.argmax(dim=1)
        losses = []
        margin = float(self.cfg["hyperedge_margin"])
        for index in range(len(labels)):
            positive = edge_class == labels[index]
            negative = edge_class == competitors[index]
            if not positive.any() or not negative.any():
                continue
            positive_score = (
                edge_scores[index][positive]
                * self.edge_distinctiveness[positive]
            ).max()
            negative_score = (
                edge_scores[index][negative]
                * self.edge_distinctiveness[negative]
            ).max()
            losses.append(
                F.softplus(margin - positive_score + negative_score)
            )
        if not losses:
            return edge_scores.new_zeros(())
        return torch.stack(losses).mean()

    def distillation_loss(self, output):
        return (
            1
            - F.cosine_similarity(
                output["global_feature"],
                output["teacher_feature"],
                dim=1,
            )
        ).mean()

    def prototype_anchor_loss(self):
        return (
            1
            - F.cosine_similarity(
                normalize(self.class_prototypes),
                self.initial_class_prototypes,
                dim=1,
            )
        ).mean()

    def graph_margin_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ):
        if self.mode not in {
            "graph_margin_atomic",
            "graph_margin_full",
        }:
            return logits.new_zeros(())
        graph = (
            self.atomic_class_graph
            if self.mode == "graph_margin_atomic"
            else self.full_class_graph
        )
        graph_rows = graph[labels]
        graph_rows = graph_rows.scatter(
            1, labels[:, None], -1.0
        )
        graph_k = min(
            int(self.cfg["graph_hard_neighbors"]),
            self.num_classes - 1,
        )
        graph_negative = graph_rows.topk(
            graph_k, dim=1
        ).indices

        visual_logits = logits.detach().clone()
        visual_logits.scatter_(1, labels[:, None], -1e4)
        visual_k = min(
            int(self.cfg["visual_hard_neighbors"]),
            self.num_classes - 1,
        )
        visual_negative = visual_logits.topk(
            visual_k, dim=1
        ).indices
        negatives = torch.cat(
            [graph_negative, visual_negative], dim=1
        )
        negative_logits = logits.gather(1, negatives)
        similarity = graph[labels[:, None], negatives]
        target_positive = self.positive_matrix[labels]
        negative_positive = self.positive_matrix[negatives]
        exclusive_count = (
            target_positive[:, None, :]
            * (1.0 - negative_positive)
        ).sum(dim=2)
        difficulty = similarity / (1.0 + exclusive_count)
        margin = (
            float(self.cfg["graph_margin_base"])
            + float(self.cfg["graph_margin_scale"]) * similarity
        )
        weights = (
            float(self.cfg["graph_weight_floor"])
            + difficulty
        )
        correct = logits.gather(
            1, labels[:, None]
        )
        losses = F.softplus(
            margin + negative_logits - correct
        )
        return (weights * losses).sum() / weights.sum().clamp_min(1e-6)

    def pseudo_symptom_losses(
        self,
        output,
        targets: torch.Tensor,
        supervision_weight: torch.Tensor,
        region_targets: torch.Tensor,
        region_mask: torch.Tensor,
    ):
        valid = (targets >= 0) & (supervision_weight > 0)
        if valid.any():
            atomic_values = F.binary_cross_entropy_with_logits(
                output["symptom_scores"][valid],
                targets[valid],
                reduction="none",
            )
            atomic = (
                atomic_values * supervision_weight[valid]
            ).sum() / supervision_weight[valid].sum().clamp_min(1e-6)
        else:
            atomic = output["symptom_scores"].new_zeros(())

        region_logits = output["region_symptom_scores"].permute(
            0, 2, 1
        )
        if region_mask.any():
            log_probability = F.log_softmax(
                region_logits[region_mask], dim=1
            )
            region = -(
                region_targets[region_mask] * log_probability
            ).sum(dim=1).mean()
        else:
            region = output["symptom_scores"].new_zeros(())

        edge = output["symptom_scores"].new_zeros(())
        if self.edge_components.shape[0] > 0:
            component_mask = self.edge_components.bool()[None, :, :]
            expanded_weight = supervision_weight[:, None, :].expand(
                -1, self.edge_components.shape[0], -1
            )
            known_components = (
                (expanded_weight > 0) | ~component_mask
            ).all(dim=2)
            expanded_target = targets.clamp(0, 1)[:, None, :].expand(
                -1, self.edge_components.shape[0], -1
            )
            # A conjunction is only as visible as its least visible atomic
            # component.
            edge_target = expanded_target.masked_fill(
                ~component_mask, 1.0
            ).min(dim=2).values
            edge_weight = expanded_weight.masked_fill(
                ~component_mask, 1.0
            ).min(dim=2).values
            edge_valid = known_components & (edge_weight > 0)
            if edge_valid.any():
                edge_score = self.hyperedge_scores(
                    output["symptom_scores"]
                )
                edge_probability = edge_score.clamp(1e-5, 1 - 1e-5)
                edge_values = F.binary_cross_entropy(
                    edge_probability[edge_valid],
                    edge_target[edge_valid],
                    reduction="none",
                )
                selected_weight = edge_weight[edge_valid]
                edge = (
                    edge_values * selected_weight
                ).sum() / selected_weight.sum().clamp_min(1e-6)
        return atomic, region, edge
