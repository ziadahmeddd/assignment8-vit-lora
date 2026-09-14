"""
Manual (from-scratch) LoRA implementation for ViTForImageClassification.

No PEFT library is used anywhere in this file. LoRA is implemented as a thin
wrapper around nn.Linear that adds a frozen base path plus a trainable
low-rank update:

    h = W0 x + b0 + (alpha / r) * B (A x)

W0, b0 stay frozen (requires_grad=False). Only A, B (and, separately, the
classification head) are trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a frozen base path + trainable LoRA path."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank  # common default: alpha == rank
        self.scaling = self.alpha / self.rank

        # Frozen base linear (keep original weights, just stop gradients).
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # Trainable low-rank factors.
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        # Standard LoRA init: A ~ Kaiming uniform, B = 0 so the adapter starts as a no-op.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """W0 + scaling * B @ A, useful for exporting a merged (deployment) checkpoint."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float | None = None  # defaults to rank if None
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("query", "value")  # attention submodule names to replace
    target_blocks: tuple[int, ...] | None = None  # None => all blocks; else e.g. (9, 10, 11)


def _resolve_target_blocks(num_layers: int, target_blocks: Iterable[int] | None) -> List[int]:
    if target_blocks is None:
        return list(range(num_layers))
    resolved = []
    for b in target_blocks:
        idx = b if b >= 0 else num_layers + b
        if not (0 <= idx < num_layers):
            raise ValueError(f"target block {b} out of range for {num_layers} layers")
        resolved.append(idx)
    return resolved


def _get_encoder_layers(model: nn.Module):
    """HF has renamed this across transformers versions:
      transformers <5:  model.vit.encoder.layer   (ModuleList)
      transformers >=5: model.vit.layers          (ModuleList)
    Support both so this works whatever version Kaggle has pinned."""
    vit = model.vit
    if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        return vit.encoder.layer
    if hasattr(vit, "layers"):
        return vit.layers
    raise AttributeError("Could not locate ViT transformer blocks on this model.")


# Each entry: name -> list of (parent_path, attr) candidates tried in order,
# to cover both the old (attention.attention.query / attention.output.dense /
# intermediate.dense / output.dense) and new (attention.q_proj / attention.o_proj /
# mlp.fc1 / mlp.fc2) transformers module layouts.
_TARGET_CANDIDATES = {
    "query": [("attention.attention", "query"), ("attention", "q_proj")],
    "key": [("attention.attention", "key"), ("attention", "k_proj")],
    "value": [("attention.attention", "value"), ("attention", "v_proj")],
    "output": [("attention.output", "dense"), ("attention", "o_proj")],
    "intermediate": [("intermediate", "dense"), ("mlp", "fc1")],
    "mlp_output": [("output", "dense"), ("mlp", "fc2")],
}


def _resolve_parent(block: nn.Module, dotted_path: str) -> nn.Module:
    obj = block
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _find_target(block: nn.Module, name: str) -> tuple[nn.Module, str]:
    if name not in _TARGET_CANDIDATES:
        raise ValueError(f"Unknown target_module '{name}'. Valid: {list(_TARGET_CANDIDATES)}")
    for parent_path, attr in _TARGET_CANDIDATES[name]:
        try:
            parent = _resolve_parent(block, parent_path)
        except AttributeError:
            continue
        if hasattr(parent, attr) and isinstance(getattr(parent, attr), nn.Linear):
            return parent, attr
    raise AttributeError(
        f"Could not locate target module '{name}' on block (tried {_TARGET_CANDIDATES[name]})."
    )


def apply_lora_to_vit(model: nn.Module, config: LoRAConfig) -> List[str]:
    """
    Replaces the chosen Linear projections inside the ViT transformer blocks with
    LoRALinear wrappers, in place. Returns the list of module paths that were replaced.

    target_modules names:
      "query", "key", "value"  -> attention Q/K/V projections
      "output"                 -> attention out-projection
      "intermediate"           -> MLP up-projection
      "mlp_output"             -> MLP down-projection
    """
    encoder_layers = _get_encoder_layers(model)
    num_layers = len(encoder_layers)
    block_indices = _resolve_target_blocks(num_layers, config.target_blocks)

    replaced: List[str] = []
    for layer_idx in block_indices:
        block = encoder_layers[layer_idx]
        for name in config.target_modules:
            parent, attr = _find_target(block, name)
            original = getattr(parent, attr)
            wrapped = LoRALinear(original, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
            setattr(parent, attr, wrapped)
            replaced.append(f"vit.layer.{layer_idx}.{name}")
    return replaced


def freeze_all_but_lora_and_head(model: nn.Module, head_attr: str = "classifier") -> None:
    """Freezes every parameter except LoRA A/B matrices and the classification head."""
    for name, param in model.named_parameters():
        is_lora = ".lora_A" in name or ".lora_B" in name
        is_head = name.startswith(f"{head_attr}.")
        param.requires_grad = is_lora or is_head


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_state_dict(model: nn.Module) -> dict:
    """State dict restricted to trainable tensors only (LoRA A/B + head) — used for
    the 'adapter-only checkpoint size' metric, which is the whole point of LoRA."""
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
