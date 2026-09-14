"""
Experiment orchestration: full fine-tuning + the manual-LoRA rank sweep
(and the bonus target-layer sweep), producing a comparison table.

This module is dataset/GPU agnostic — it's imported both by a local dry run
(tiny random data, CPU) and by the Kaggle notebook (real EuroSAT, GPU).
"""
from __future__ import annotations

import copy
from typing import Callable, List

import torch
from transformers import ViTForImageClassification

from lora import (
    LoRAConfig,
    apply_lora_to_vit,
    freeze_all_but_lora_and_head,
    trainable_state_dict,
)
from metrics import RunResult, train_model, state_dict_size_mb


def full_state_dict_mb(model: torch.nn.Module) -> float:
    return state_dict_size_mb({k: v.cpu() for k, v in model.state_dict().items()})


def build_full_finetune_model(model_name: str, num_labels: int, id2label: dict, label2id: dict):
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    for p in model.parameters():
        p.requires_grad = True
    return model


def build_lora_model(model_name: str, num_labels: int, id2label: dict, label2id: dict, lora_cfg: LoRAConfig):
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    replaced = apply_lora_to_vit(model, lora_cfg)
    freeze_all_but_lora_and_head(model)
    return model, replaced


def run_full_finetune(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float = 5e-5,
) -> RunResult:
    model = build_full_finetune_model(model_name, num_labels, id2label, label2id).to(device)
    result = train_model(
        name="full_finetune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        trainable_state_dict_fn=lambda m: {k: v.cpu() for k, v in m.state_dict().items()},
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_lora_experiment(
    run_name: str,
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lora_cfg: LoRAConfig,
    lr: float = 1e-3,
) -> RunResult:
    model, replaced = build_lora_model(model_name, num_labels, id2label, label2id, lora_cfg)
    model.to(device)
    print(f"[{run_name}] LoRA applied to: {replaced}")
    result = train_model(
        name=run_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        trainable_state_dict_fn=trainable_state_dict,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_rank_sweep(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    ranks: List[int] = (16, 32, 64),
    target_modules=("query", "value"),
    target_blocks=(-3, -2, -1),  # last three transformer blocks
    lr: float = 1e-3,
) -> List[RunResult]:
    results = []
    for rank in ranks:
        cfg = LoRAConfig(rank=rank, target_modules=target_modules, target_blocks=target_blocks)
        res = run_lora_experiment(
            run_name=f"lora_r{rank}_QV_last3blocks",
            model_name=model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lora_cfg=cfg,
            lr=lr,
        )
        results.append(res)
    return results


def run_target_layer_sweep(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    best_rank: int,
    target_blocks=(-3, -2, -1),
    lr: float = 1e-3,
) -> List[RunResult]:
    """Bonus sweep (objectives list explicitly calls out 'Different target layers')."""
    variants = {
        "QV": ("query", "value"),
        "QKV": ("query", "key", "value"),
        "QKVO": ("query", "key", "value", "output"),
        "QV+MLP": ("query", "value", "intermediate", "mlp_output"),
    }
    results = []
    for variant_name, modules in variants.items():
        cfg = LoRAConfig(rank=best_rank, target_modules=modules, target_blocks=target_blocks)
        res = run_lora_experiment(
            run_name=f"lora_r{best_rank}_{variant_name}_last3blocks",
            model_name=model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lora_cfg=cfg,
            lr=lr,
        )
        results.append(res)
    return results
