from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import normalize
from .finetune_model import SymptomGuidedCLIP


SEMANTIC_GROUPS = {
    "organ": {
        "affected_organ", "organ_shape", "organ_margin",
    },
    "appearance": {
        "color", "lesion_shape", "lesion_boundary",
    },
    "texture": {
        "lesion_texture", "surface_texture", "pathogen_sign",
    },
    "change": {
        "tissue_change", "progression", "spatial_pattern",
    },
}


def _kmeans(features: torch.Tensor, clusters: int, iterations: int):
    features = normalize(features)
    if len(features) == 0:
        raise ValueError("Cannot initialize prototypes from an empty class")
    if len(features) < clusters:
        repeat = math.ceil(clusters / len(features))
        return features.repeat(repeat, 1)[:clusters].clone()

    # Deterministic farthest-point initialization.
    centers = [features[0]]
    for _ in range(1, clusters):
        similarity = features @ torch.stack(centers).t()
        distance = 1.0 - similarity.max(dim=1).values
        centers.append(features[distance.argmax()])
    centers = torch.stack(centers)
    for _ in range(iterations):
        assignment = (features @ centers.t()).argmax(dim=1)
        updated = []
        for cluster_id in range(clusters):
            selected = features[assignment == cluster_id]
            if len(selected):
                updated.append(normalize(selected.mean(dim=0)))
            else:
                updated.append(centers[cluster_id])
        updated = torch.stack(updated)
        if torch.allclose(updated, centers, atol=1e-5):
            break
        centers = updated
    return normalize(centers)


class SymptomConstrainedMultiPrototypeCLIP(SymptomGuidedCLIP):
    """Visual-text multi-prototype distribution learning for PlantWild."""

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
        super().__init__(
            clip_model,
            class_text_features,
            class_text_labels,
            symptom_features,
            graph,
            cfg,
            mode="class_only",
        )
        if mode not in {"visual", "semantic", "full"}:
            raise ValueError("Unsupported multi-prototype mode: %s" % mode)
        self.multiprototype_mode = mode
        self.num_visual_prototypes = int(cfg["num_visual_prototypes"])
        dim = symptom_features.shape[1]

        initial_class = self.class_mean_prototypes().detach()
        self.visual_prototypes = nn.Parameter(
            initial_class[:, None, :].repeat(
                1, self.num_visual_prototypes, 1
            )
        )

        semantic, semantic_mask, group_names = (
            self._build_semantic_prototypes(
                graph, symptom_features, initial_class
            )
        )
        self.semantic_group_names = group_names
        self.register_buffer(
            "initial_semantic_prototypes", semantic
        )
        self.register_buffer("semantic_group_mask", semantic_mask)
        self.semantic_adapter = nn.Sequential(
            nn.Linear(dim, int(cfg["semantic_hidden"]), bias=False),
            nn.GELU(),
            nn.Dropout(float(cfg["prototype_dropout"])),
            nn.Linear(int(cfg["semantic_hidden"]), dim, bias=False),
        )
        nn.init.zeros_(self.semantic_adapter[-1].weight)
        # The adapter output starts at zero, while a small non-zero scale
        # preserves gradients into its final layer.
        self.semantic_adapter_scale = nn.Parameter(torch.tensor(0.10))

        router_hidden = int(cfg["router_hidden"])
        router_input = dim + 4
        self.visual_router = nn.Sequential(
            nn.Linear(router_input, router_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg["prototype_dropout"])),
            nn.Linear(router_hidden, 1),
        )
        self.semantic_router = nn.Sequential(
            nn.Linear(router_input, router_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg["prototype_dropout"])),
            nn.Linear(router_hidden, 1),
        )
        nn.init.zeros_(self.visual_router[-1].weight)
        nn.init.zeros_(self.semantic_router[-1].weight)
        nn.init.constant_(
            self.visual_router[-1].bias,
            float(cfg["router_initial_bias"]),
        )
        nn.init.constant_(
            self.semantic_router[-1].bias,
            float(cfg["router_initial_bias"]),
        )

        self.visual_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(cfg["visual_logit_scale"])))
        )
        self.semantic_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(cfg["semantic_logit_scale"])))
        )
        self.visual_route_temperature = nn.Parameter(
            torch.tensor(
                math.log(float(cfg["visual_route_temperature"]))
            )
        )
        self.semantic_route_temperature = nn.Parameter(
            torch.tensor(
                math.log(float(cfg["semantic_route_temperature"]))
            )
        )
        self.max_visual_gate = float(cfg["max_visual_gate"])
        self.max_semantic_gate = float(cfg["max_semantic_gate"])

        # Earlier symptom heads are intentionally inactive in this model.
        for parameter in self.symptom_projection.parameters():
            parameter.requires_grad_(False)
        self.symptom_projection_scale.requires_grad_(False)
        self.symptom_logit_scale.requires_grad_(False)
        self.symptom_logit_bias.requires_grad_(False)
        self.latent_strength_raw.requires_grad_(False)

    @staticmethod
    def _build_semantic_prototypes(
        graph: dict,
        symptom_features: torch.Tensor,
        fallback: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        group_names = list(SEMANTIC_GROUPS)
        positive = torch.tensor(
            graph["positive_matrix"], dtype=torch.bool
        )
        idf = torch.tensor([
            float(item.get("idf", 1.0))
            for item in graph["symptoms"]
        ])
        symptom_group = []
        for item in graph["symptoms"]:
            assigned = -1
            for group_id, group_name in enumerate(group_names):
                if item["attribute_type"] in SEMANTIC_GROUPS[group_name]:
                    assigned = group_id
                    break
            symptom_group.append(assigned)

        prototypes = torch.zeros(
            positive.shape[0], len(group_names), symptom_features.shape[1]
        )
        mask = torch.zeros(
            positive.shape[0], len(group_names), dtype=torch.bool
        )
        for class_id in range(positive.shape[0]):
            for group_id in range(len(group_names)):
                selected = positive[class_id].clone()
                selected &= torch.tensor([
                    value == group_id for value in symptom_group
                ])
                if selected.any():
                    weights = idf[selected]
                    value = (
                        symptom_features[selected] * weights[:, None]
                    ).sum(dim=0) / weights.sum().clamp_min(1e-6)
                    prototypes[class_id, group_id] = normalize(value)
                    mask[class_id, group_id] = True
                else:
                    prototypes[class_id, group_id] = fallback[class_id]
        return normalize(prototypes), mask, group_names

    @torch.no_grad()
    def initialize_visual_prototypes(
        self,
        loader,
        device: torch.device,
        iterations: int,
    ):
        self.eval()
        by_class = [[] for _ in range(self.num_classes)]
        for images, labels, _ in loader:
            global_images = images[:, 0].to(device, non_blocking=True)
            features = normalize(self.student.encode_image(global_images))
            for feature, label in zip(features.cpu(), labels.tolist()):
                by_class[label].append(feature)
        centers = []
        for class_id, values in enumerate(by_class):
            if values:
                class_features = torch.stack(values)
                centers.append(_kmeans(
                    class_features,
                    self.num_visual_prototypes,
                    iterations,
                ))
            else:
                centers.append(
                    self.class_mean_prototypes()[class_id][None].repeat(
                        self.num_visual_prototypes, 1
                    ).cpu()
                )
        self.visual_prototypes.copy_(
            normalize(torch.stack(centers)).to(
                self.visual_prototypes.device
            )
        )

    def trainable_groups(self):
        backbone = list(self.student.visual.layer4.parameters())
        backbone += list(self.student.visual.attnpool.parameters())
        base_head = [self.class_prototypes, self.logit_scale]
        novel = [
            self.visual_prototypes,
            self.semantic_adapter_scale,
            self.visual_logit_scale,
            self.semantic_logit_scale,
            self.visual_route_temperature,
            self.semantic_route_temperature,
        ]
        novel += list(self.semantic_adapter.parameters())
        novel += list(self.visual_router.parameters())
        novel += list(self.semantic_router.parameters())
        return (
            list(dict.fromkeys(backbone)),
            list(dict.fromkeys(base_head)),
            list(dict.fromkeys(novel)),
        )

    def semantic_prototypes(self):
        residual = self.semantic_adapter(
            self.initial_semantic_prototypes
        )
        return normalize(
            self.initial_semantic_prototypes
            + self.semantic_adapter_scale.tanh() * residual
        )

    @staticmethod
    def _confidence_features(logits):
        probability = logits.softmax(dim=1)
        top2 = probability.topk(2, dim=1).values
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        entropy /= math.log(logits.shape[1])
        return torch.cat([
            top2[:, :1],
            top2[:, 1:2],
            top2[:, :1] - top2[:, 1:2],
            entropy,
        ], dim=1)

    def _global_logits(self, feature):
        scale = self.logit_scale.exp().clamp(1, 100)
        description_logits = scale * feature @ normalize(
            self.class_prototypes
        ).t()
        text_mean, text_max = self.aggregate_class_descriptions(
            description_logits
        )
        logits = (
            float(self.cfg["description_mean_weight"]) * text_mean
            + float(self.cfg["description_max_weight"]) * text_max
        )
        return logits, text_mean, text_max

    def visual_branch(self, feature):
        prototypes = normalize(self.visual_prototypes)
        similarity = torch.einsum(
            "bd,ckd->bck", feature, prototypes
        )
        temperature = self.visual_route_temperature.exp().clamp(
            0.02, 2.0
        )
        assignment = F.softmax(similarity / temperature, dim=2)
        score = (assignment * similarity).sum(dim=2)
        logits = self.visual_logit_scale.exp().clamp(1, 100) * score
        return logits, assignment

    def semantic_branch(self, feature):
        prototypes = self.semantic_prototypes()
        similarity = torch.einsum(
            "bd,cgd->bcg", feature, prototypes
        )
        mask = self.semantic_group_mask[None]
        temperature = self.semantic_route_temperature.exp().clamp(
            0.02, 2.0
        )
        route_logits = (similarity / temperature).masked_fill(
            ~mask, -1e4
        )
        assignment = F.softmax(route_logits, dim=2)
        score = (assignment * similarity).sum(dim=2)
        logits = self.semantic_logit_scale.exp().clamp(1, 100) * score
        return logits, assignment

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, regions, channels, height, width = images.shape
        flattened = images.reshape(
            batch * regions, channels, height, width
        )
        encoded = normalize(self.student.encode_image(flattened))
        encoded = encoded.reshape(batch, regions, -1)
        feature = encoded[:, 0]
        base_logits, text_mean, text_max = self._global_logits(feature)
        visual_logits, visual_assignment = self.visual_branch(feature)
        semantic_logits, semantic_assignment = self.semantic_branch(feature)

        router_input = torch.cat([
            feature, self._confidence_features(base_logits)
        ], dim=1)
        visual_gate = self.max_visual_gate * torch.sigmoid(
            self.visual_router(router_input)
        )
        semantic_gate = self.max_semantic_gate * torch.sigmoid(
            self.semantic_router(router_input)
        )
        if self.multiprototype_mode == "semantic":
            visual_gate = visual_gate * 0
        if self.multiprototype_mode == "visual":
            semantic_gate = semantic_gate * 0

        logits = (
            base_logits
            + visual_gate * (visual_logits - base_logits)
            + semantic_gate * (semantic_logits - base_logits)
        )
        with torch.no_grad():
            teacher_feature = self.encode_teacher(images[:, 0])
        zeros = torch.zeros_like(base_logits)
        return {
            "logits": logits,
            "base_logits": base_logits,
            "visual_logits": visual_logits,
            "semantic_logits": semantic_logits,
            "text_mean_logits": text_mean,
            "text_max_logits": text_max,
            "global_feature": feature,
            "teacher_feature": teacher_feature,
            "visual_assignment": visual_assignment,
            "semantic_assignment": semantic_assignment,
            "visual_gate": visual_gate,
            "semantic_gate": semantic_gate,
            "symptom_scores": semantic_logits,
            "hyperedge_scores": base_logits.new_zeros(batch, 0),
            "latent_correction": zeros,
        }

    def visual_diversity_loss(self):
        prototypes = normalize(self.visual_prototypes)
        similarity = torch.einsum(
            "ckd,cjd->ckj", prototypes, prototypes
        )
        eye = torch.eye(
            self.num_visual_prototypes,
            device=similarity.device,
            dtype=torch.bool,
        )[None]
        off_diagonal = similarity.masked_select(~eye)
        margin = float(self.cfg["visual_diversity_margin"])
        return F.relu(off_diagonal - margin).mean()

    def distribution_alignment_loss(self):
        visual = normalize(self.visual_prototypes)
        semantic = self.semantic_prototypes()
        similarity = torch.einsum(
            "ckd,cgd->ckg", visual, semantic
        )
        similarity = similarity.masked_fill(
            ~self.semantic_group_mask[:, None, :], -1e4
        )
        temperature = float(self.cfg["alignment_temperature"])
        soft_match = temperature * torch.logsumexp(
            similarity / temperature, dim=2
        )
        soft_match -= temperature * torch.log(
            self.semantic_group_mask.sum(dim=1)
            .clamp_min(1).float()
        )[:, None]
        return (1.0 - soft_match).mean()

    def route_entropy_loss(self, output):
        values: Iterable[torch.Tensor] = []
        if self.multiprototype_mode in {"visual", "full"}:
            values = list(values) + [output["visual_assignment"]]
        if self.multiprototype_mode in {"semantic", "full"}:
            values = list(values) + [output["semantic_assignment"]]
        losses = []
        for assignment in values:
            entropy = -(
                assignment * assignment.clamp_min(1e-8).log()
            ).sum(dim=2)
            losses.append(entropy.mean())
        return torch.stack(losses).mean()
