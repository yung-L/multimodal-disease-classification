from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import normalize


class PairwiseSymptomCorrector(nn.Module):
    """Bounded correction for validation-observed confusion pairs."""

    def __init__(
        self,
        base_model,
        pair_sources: torch.Tensor,
        pair_targets: torch.Tensor,
        pair_features: torch.Tensor,
        cfg: dict,
    ):
        super().__init__()
        self.base = base_model
        self.cfg = cfg
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.register_buffer("pair_sources", pair_sources.long())
        self.register_buffer("pair_targets", pair_targets.long())
        self.register_buffer(
            "pair_features", normalize(pair_features)
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(
                pair_features.shape[1],
                int(cfg["pair_hidden"]),
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(
                int(cfg["pair_hidden"]),
                pair_features.shape[1],
                bias=False,
            ),
        )
        nn.init.zeros_(self.pair_projection[-1].weight)
        self.pair_projection_scale = nn.Parameter(
            torch.tensor(float(cfg["pair_projection_scale"]))
        )
        self.correction_strength_raw = nn.Parameter(
            torch.tensor(float(cfg["correction_strength_init"]))
        )
        self.num_classes = base_model.num_classes

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        if mode:
            self.pair_projection.train()
        return self

    def pair_evidence(self, regions: torch.Tensor):
        features = normalize(
            self.pair_features
            + self.pair_projection_scale
            * self.pair_projection(self.pair_features)
        )
        similarity = torch.einsum(
            "brd,pd->brp", regions, features
        )
        temperature = float(self.cfg["region_temperature"])
        return temperature * torch.logsumexp(
            similarity / temperature, dim=1
        )

    def forward(self, images: torch.Tensor):
        with torch.no_grad():
            regions = self.base.encode_student(images)
            global_feature = regions[:, 0]
            scale = self.base.logit_scale.exp().clamp(1, 100)
            description_logits = (
                scale
                * global_feature
                @ normalize(self.base.class_prototypes).t()
            )
            text_mean, text_max = (
                self.base.aggregate_class_descriptions(
                    description_logits
                )
            )
            base_logits = (
                float(self.base.cfg["description_mean_weight"])
                * text_mean
                + float(self.base.cfg["description_max_weight"])
                * text_max
            )
        evidence = self.pair_evidence(regions)
        batch = len(images)
        correction = base_logits.new_zeros(base_logits.shape)

        topk = min(
            int(self.cfg["candidate_topk"]), self.num_classes
        )
        candidate_ids = base_logits.topk(topk, dim=1).indices
        source_is_candidate = (
            candidate_ids[:, :, None]
            == self.pair_sources[None, None, :]
        ).any(dim=1)
        target_is_candidate = (
            candidate_ids[:, :, None]
            == self.pair_targets[None, None, :]
        ).any(dim=1)
        active = source_is_candidate & target_is_candidate

        confidence = base_logits.softmax(dim=1).max(dim=1).values
        uncertainty = (
            (
                float(self.cfg["confidence_threshold"])
                - confidence
            )
            / float(self.cfg["confidence_temperature"])
        ).sigmoid()
        signed = evidence * active.float() * uncertainty[:, None]
        correction.scatter_add_(
            1,
            self.pair_sources[None, :].expand(batch, -1),
            signed,
        )
        normalizer = active.float().sum(dim=1, keepdim=True).clamp_min(1)
        correction = correction / normalizer
        strength = (
            float(self.cfg["max_correction"])
            * self.correction_strength_raw.sigmoid()
        )
        return {
            "logits": base_logits + strength * correction,
            "base_logits": base_logits,
            "pair_evidence": evidence,
            "active_pairs": active,
            "correction": strength * correction,
            "strength": strength,
            "candidate_ids": candidate_ids,
        }

    def pairwise_loss(self, output, labels):
        pair_logits = output["pair_evidence"]
        source_match = (
            self.pair_sources[None, :] == labels[:, None]
        )
        active = source_match & output["active_pairs"]
        if not active.any():
            return pair_logits.new_zeros(())
        competitor_source = (
            self.pair_sources[None, :]
            == output["base_logits"].detach()
            .masked_fill(
                F.one_hot(
                    labels, self.num_classes
                ).bool(),
                -1e4,
            )
            .argmax(dim=1)[:, None]
        )
        reverse = (
            competitor_source
            & (
                self.pair_targets[None, :]
                == labels[:, None]
            )
        )
        positive = pair_logits.masked_fill(~active, -1e4).max(dim=1).values
        negative = pair_logits.masked_fill(~reverse, -1e4).max(dim=1).values
        valid = active.any(dim=1) & reverse.any(dim=1)
        if not valid.any():
            return pair_logits.new_zeros(())
        return F.softplus(
            float(self.cfg["pair_margin"])
            - positive[valid]
            + negative[valid]
        ).mean()
