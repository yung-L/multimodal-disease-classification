from __future__ import annotations

import copy

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from s2dpg.common import normalize


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(prompts.dtype)
        eot = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, cfg):
        super().__init__()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.num_classes = len(self.classnames)
        self.class_specific = bool(cfg["class_specific_context"])
        self.class_token_position = str(cfg["class_token_position"])
        context_init = str(cfg.get("context_init", "")).strip()
        dtype = clip_model.dtype
        context_dim = int(clip_model.ln_final.weight.shape[0])

        if context_init:
            context_init = context_init.replace("_", " ")
            n_ctx = len(context_init.split())
            tokenized = clip.tokenize(context_init).to(
                clip_model.positional_embedding.device
            )
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized).to(dtype)
            initial = embedding[0, 1 : 1 + n_ctx, :]
            prefix_text = context_init
        else:
            n_ctx = int(cfg["n_ctx"])
            shape = (
                (self.num_classes, n_ctx, context_dim)
                if self.class_specific else (n_ctx, context_dim)
            )
            initial = torch.empty(shape, dtype=dtype)
            nn.init.normal_(initial, std=0.02)
            prefix_text = " ".join(["X"] * n_ctx)

        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(initial)
        prompts = [prefix_text + " " + name + "." for name in self.classnames]
        tokenized_prompts = torch.cat(
            [clip.tokenize(text) for text in prompts]
        ).to(clip_model.positional_embedding.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lengths = [
            len(clip.tokenize(name)[0].nonzero()) - 2 for name in self.classnames
        ]

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)
        if self.class_token_position == "end":
            return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        prompts = []
        half = self.n_ctx // 2
        for class_id, name_length in enumerate(self.name_lengths):
            prefix = self.token_prefix[class_id : class_id + 1]
            class_tokens = self.token_suffix[
                class_id : class_id + 1, :name_length, :
            ]
            suffix = self.token_suffix[
                class_id : class_id + 1, name_length:, :
            ]
            current = ctx[class_id : class_id + 1]
            if self.class_token_position == "front":
                prompt = torch.cat([prefix, class_tokens, current, suffix], dim=1)
            elif self.class_token_position == "middle":
                prompt = torch.cat([
                    prefix, current[:, :half], class_tokens,
                    current[:, half:], suffix,
                ], dim=1)
            else:
                raise ValueError(
                    "class_token_position must be end, middle, or front"
                )
            prompts.append(prompt)
        return torch.cat(prompts, dim=0)


class CoOpClassifier(nn.Module):
    def __init__(self, clip_model, classnames, cfg):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, cfg)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.logit_scale.requires_grad_(False)

    def text_features(self):
        prompts = self.prompt_learner()
        tokens = self.prompt_learner.tokenized_prompts
        return normalize(self.text_encoder(prompts, tokens))

    def forward(self, image_features):
        image_features = normalize(image_features)
        text_features = self.text_features()
        return self.logit_scale.exp().clamp(max=100.0) * image_features @ text_features.t()


@torch.inference_mode()
def predict(model, payload, device, batch_size):
    loader = DataLoader(
        TensorDataset(payload["features"], payload["labels"]),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    text_features = model.text_features()
    scale = model.logit_scale.exp().clamp(max=100.0)
    logits, labels = [], []
    for features, target in loader:
        features = normalize(features.to(device))
        logits.append((scale * features @ text_features.t()).cpu())
        labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def train_coop(model, cache, cfg, device, metric_fn, seed):
    model = model.to(device)
    parameters = list(model.prompt_learner.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(cfg["learning_rate"]),
        momentum=0.9,
        weight_decay=float(cfg["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(cache["train"]["features"], cache["train"]["labels"]),
        batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(cfg["epochs"]) * len(loader))
    )
    best_state, best_score, best_epoch = None, -float("inf"), 0
    stale, history = 0, []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total, count = 0.0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(
                logits, labels,
                label_smoothing=float(cfg["label_smoothing"]),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(labels)
            count += len(labels)
        val_logits, val_labels = predict(
            model, cache["val"], device, int(cfg["batch_size"])
        )
        val_metrics, _ = metric_fn(val_logits, val_labels)
        score = float(val_metrics["macro_f1"])
        row = {
            "epoch": epoch, "train_loss": total / max(count, 1),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": score,
        }
        history.append(row)
        print(
            "Epoch %03d loss=%.4f val_acc=%.2f val_f1=%.2f"
            % (epoch, row["train_loss"], row["val_accuracy"], score)
        )
        if score > best_score + 1e-8:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.prompt_learner.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("CoOp produced no checkpoint")
    model.prompt_learner.load_state_dict(best_state)
    return model, history, best_epoch, best_score
