from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import normalize
from .finetune_model import SymptomGuidedCLIP


class HierarchicalLocalGlobalCLIP(SymptomGuidedCLIP):
    """Hierarchical symptom queries over multi-scale CLIP visual tokens."""

    def __init__(
        self,
        clip_model,
        class_text_features: torch.Tensor,
        class_text_labels: torch.Tensor,
        symptom_features: torch.Tensor,
        graph: dict,
        cfg: dict,
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
        local_dim = int(cfg["local_dim"])
        heads = int(cfg["local_attention_heads"])
        if local_dim % heads:
            raise ValueError("local_dim must be divisible by attention heads")

        layer3_channels = self.student.visual.layer3[-1].conv3.out_channels
        layer4_channels = self.student.visual.layer4[-1].conv3.out_channels
        text_dim = symptom_features.shape[1]
        self.layer3_projection = nn.Conv2d(
            layer3_channels, local_dim, kernel_size=1, bias=False
        )
        self.layer4_projection = nn.Conv2d(
            layer4_channels, local_dim, kernel_size=1, bias=False
        )
        self.symptom_text_projection = nn.Linear(
            text_dim, local_dim, bias=False
        )
        if text_dim == local_dim:
            nn.init.eye_(self.symptom_text_projection.weight)
        self.initialize_local_from_backbone()

        attributes = sorted({
            item["attribute_type"] for item in graph["symptoms"]
        })
        entities = sorted({
            item["entity"] for item in graph["symptoms"]
        })
        attribute_index = {value: index for index, value in enumerate(attributes)}
        entity_index = {value: index for index, value in enumerate(entities)}
        self.register_buffer(
            "symptom_attribute_ids",
            torch.tensor([
                attribute_index[item["attribute_type"]]
                for item in graph["symptoms"]
            ], dtype=torch.long),
        )
        self.register_buffer(
            "symptom_entity_ids",
            torch.tensor([
                entity_index[item["entity"]]
                for item in graph["symptoms"]
            ], dtype=torch.long),
        )
        self.attribute_embedding = nn.Embedding(len(attributes), local_dim)
        self.entity_embedding = nn.Embedding(len(entities), local_dim)
        nn.init.zeros_(self.attribute_embedding.weight)
        nn.init.zeros_(self.entity_embedding.weight)

        hidden = int(cfg["hierarchy_hidden"])
        self.query_adapter = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(cfg["local_dropout"])),
            nn.Linear(hidden, local_dim),
        )
        nn.init.zeros_(self.query_adapter[-1].weight)
        nn.init.zeros_(self.query_adapter[-1].bias)

        self.local_attention = nn.MultiheadAttention(
            local_dim,
            heads,
            dropout=float(cfg["local_dropout"]),
            batch_first=True,
        )
        self.local_norm = nn.LayerNorm(local_dim)
        self.null_visual_token = nn.Parameter(torch.zeros(1, 1, local_dim))
        nn.init.normal_(self.null_visual_token, std=0.02)

        self.local_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(cfg["local_init_logit_scale"])))
        )
        self.local_logit_bias = nn.Parameter(torch.zeros(()))
        gate_hidden = int(cfg["gate_hidden"])
        self.fusion_gate = nn.Sequential(
            nn.Linear(text_dim + 3, gate_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg["local_dropout"])),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(
            self.fusion_gate[-1].bias,
            float(cfg["gate_initial_bias"]),
        )

        self.local_topk = int(cfg["local_symptom_topk"])
        self.max_gate = float(cfg["max_local_gate"])
        self.register_buffer(
            "class_symptom_mask", self.positive_matrix.bool()
        )
        # These heads belong to earlier experiments and are not used by the
        # hierarchical local branch.
        for parameter in self.symptom_projection.parameters():
            parameter.requires_grad_(False)
        self.symptom_projection_scale.requires_grad_(False)
        self.symptom_logit_scale.requires_grad_(False)
        self.symptom_logit_bias.requires_grad_(False)
        self.latent_strength_raw.requires_grad_(False)

    def initialize_local_from_backbone(self):
        if (
            hasattr(self.student.visual.attnpool, "c_proj")
            and self.student.visual.attnpool.c_proj.weight.shape
            == self.layer4_projection.weight[:, :, 0, 0].shape
        ):
            with torch.no_grad():
                self.layer4_projection.weight[:, :, 0, 0].copy_(
                    self.student.visual.attnpool.c_proj.weight
                )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.layer3_projection.train()
            self.layer4_projection.train()
            self.symptom_text_projection.train()
            self.attribute_embedding.train()
            self.entity_embedding.train()
            self.query_adapter.train()
            self.local_attention.train()
            self.local_norm.train()
            self.fusion_gate.train()
            if self.cfg.get("unfreeze_layer3_last_block", True):
                self.student.visual.layer3[-1].train()
        return self

    def trainable_groups(self):
        backbone = list(self.student.visual.layer4.parameters())
        backbone += list(self.student.visual.attnpool.parameters())
        if self.cfg.get("unfreeze_layer3_last_block", True):
            for parameter in self.student.visual.layer3[-1].parameters():
                parameter.requires_grad_(True)
            backbone += list(self.student.visual.layer3[-1].parameters())
        head = [self.class_prototypes, self.logit_scale]
        local_modules = [
            self.layer3_projection,
            self.layer4_projection,
            self.symptom_text_projection,
            self.attribute_embedding,
            self.entity_embedding,
            self.query_adapter,
            self.local_attention,
            self.local_norm,
            self.fusion_gate,
        ]
        for module in local_modules:
            head += list(module.parameters())
        head += [
            self.null_visual_token,
            self.local_logit_scale,
            self.local_logit_bias,
        ]
        # Remove duplicates while preserving parameter order.
        backbone = list(dict.fromkeys(backbone))
        head = list(dict.fromkeys(head))
        return backbone, head

    @staticmethod
    def _stem(visual, images):
        for conv, bn in [
            (visual.conv1, visual.bn1),
            (visual.conv2, visual.bn2),
            (visual.conv3, visual.bn3),
        ]:
            images = visual.relu(bn(conv(images)))
        return visual.avgpool(images)

    def visual_pyramid(self, images: torch.Tensor):
        visual = self.student.visual
        x = images.to(visual.conv1.weight.dtype)
        x = self._stem(visual, x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        layer3 = visual.layer3(x)
        layer4 = visual.layer4(layer3)
        global_feature = normalize(visual.attnpool(layer4))
        return layer3, layer4, global_feature

    def hierarchy_queries(self):
        queries = self.symptom_text_projection(self.symptom_features)
        queries = (
            queries
            + self.attribute_embedding(self.symptom_attribute_ids)
            + self.entity_embedding(self.symptom_entity_ids)
        )
        return normalize(queries + self.query_adapter(queries))

    def local_branch(self, layer3, layer4, global_feature):
        tokens3 = self.layer3_projection(layer3).flatten(2).transpose(1, 2)
        tokens4 = self.layer4_projection(layer4).flatten(2).transpose(1, 2)
        visual_tokens = torch.cat([tokens3, tokens4], dim=1)
        visual_tokens = normalize(visual_tokens)
        null = self.null_visual_token.expand(len(layer3), -1, -1)
        keys = torch.cat([visual_tokens, null], dim=1)

        queries = self.hierarchy_queries()
        batch_queries = queries[None].expand(len(layer3), -1, -1)
        attended, attention = self.local_attention(
            batch_queries,
            keys,
            keys,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = normalize(self.local_norm(attended + batch_queries))
        null_probability = attention[:, :, -1].clamp(0, 1)
        reliability = 1.0 - null_probability
        alignment = (attended * batch_queries).sum(dim=-1)
        symptom_evidence = alignment * reliability

        expanded = symptom_evidence[:, None, :].expand(
            -1, self.num_classes, -1
        )
        masked = expanded.masked_fill(
            ~self.class_symptom_mask[None], -1e4
        )
        topk = min(self.local_topk, self.num_symptoms)
        selected = masked.topk(topk, dim=2).values
        valid = selected > -1e3
        pooled = selected.masked_fill(~valid, 0).sum(dim=2)
        pooled = pooled / valid.sum(dim=2).clamp_min(1)
        local_logits = (
            self.local_logit_scale.exp().clamp(1, 100) * pooled
            + self.local_logit_bias
        )

        evidence_probability = torch.sigmoid(
            symptom_evidence
            / float(self.cfg["local_evidence_temperature"])
        )
        top_evidence = evidence_probability.topk(
            min(3, self.num_symptoms), dim=1
        ).values.mean(dim=1, keepdim=True)
        mean_reliability = reliability.mean(dim=1, keepdim=True)
        attention_concentration = attention[:, :, :-1].amax(dim=2).mean(
            dim=1, keepdim=True
        )
        gate_input = torch.cat([
            global_feature,
            top_evidence,
            mean_reliability,
            attention_concentration,
        ], dim=1)
        gate = self.max_gate * torch.sigmoid(self.fusion_gate(gate_input))
        return {
            "local_logits": local_logits,
            "symptom_evidence": symptom_evidence,
            "symptom_reliability": reliability,
            "local_attention": attention[:, :, :-1],
            "null_attention": null_probability,
            "fusion_gate": gate,
        }

    def _global_logits(self, global_feature):
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
        return base_logits, text_mean, text_max

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_images = images[:, 0]
        layer3, layer4, global_feature = self.visual_pyramid(global_images)
        base_logits, text_mean, text_max = self._global_logits(global_feature)
        local = self.local_branch(layer3, layer4, global_feature)
        logits = base_logits + local["fusion_gate"] * (
            local["local_logits"] - base_logits
        )
        with torch.no_grad():
            teacher_feature = self.encode_teacher(global_images)
        zeros = torch.zeros_like(base_logits)
        return {
            "logits": logits,
            "base_logits": base_logits,
            "local_logits": local["local_logits"],
            "text_mean_logits": text_mean,
            "text_max_logits": text_max,
            "global_feature": global_feature,
            "teacher_feature": teacher_feature,
            "symptom_scores": local["symptom_evidence"],
            "symptom_reliability": local["symptom_reliability"],
            "local_attention": local["local_attention"],
            "null_attention": local["null_attention"],
            "fusion_gate": local["fusion_gate"],
            "hyperedge_scores": base_logits.new_zeros(
                len(base_logits), 0
            ),
            "latent_correction": zeros,
        }

    def local_consistency_loss(self, output):
        temperature = float(self.cfg["consistency_temperature"])
        teacher = F.softmax(
            output["base_logits"].detach() / temperature, dim=1
        )
        student = F.log_softmax(
            output["local_logits"] / temperature, dim=1
        )
        return F.kl_div(
            student, teacher, reduction="batchmean"
        ) * temperature * temperature

    def attention_sparsity_loss(self, output):
        attention = output["local_attention"].clamp_min(1e-8)
        entropy = -(attention * attention.log()).sum(dim=2)
        normalizer = math.log(max(attention.shape[2], 2))
        return (entropy / normalizer).mean()

    def gate_regularization_loss(self, output):
        return output["fusion_gate"].mean()
