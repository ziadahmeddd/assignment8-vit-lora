"""Metric/utility helpers shared by every experiment (full FT + all LoRA runs)."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def state_dict_size_mb(state_dict: dict) -> float:
    """Size in MB of a state dict if it were saved to disk (torch.save, no compression)."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


class GPUMemoryTracker:
    """Peak CUDA memory allocated during a `with` block. No-ops gracefully on CPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_mb = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return False


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1_macro: float
    seconds: float


@dataclass
class RunResult:
    name: str
    trainable_params: int
    total_params: int
    checkpoint_size_mb: float
    epochs: List[EpochLog] = field(default_factory=list)
    total_train_seconds: float = 0.0
    peak_gpu_mem_mb: float = 0.0

    @property
    def final_val_accuracy(self) -> float:
        return self.epochs[-1].val_accuracy if self.epochs else float("nan")

    @property
    def final_val_f1(self) -> float:
        return self.epochs[-1].val_f1_macro if self.epochs else float("nan")

    @property
    def final_train_loss(self) -> float:
        return self.epochs[-1].train_loss if self.epochs else float("nan")

    @property
    def final_val_loss(self) -> float:
        return self.epochs[-1].val_loss if self.epochs else float("nan")

    def as_row(self) -> dict:
        return {
            "run": self.name,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_%": round(100 * self.trainable_params / self.total_params, 4),
            "checkpoint_MB": round(self.checkpoint_size_mb, 2),
            "train_loss": round(self.final_train_loss, 4),
            "val_loss": round(self.final_val_loss, 4),
            "val_accuracy": round(self.final_val_accuracy, 4),
            "val_f1_macro": round(self.final_val_f1, 4),
            "train_seconds": round(self.total_train_seconds, 1),
            "peak_gpu_mem_MB": round(self.peak_gpu_mem_mb, 1),
        }


def run_one_epoch(model, loader, optimizer, device, criterion, train: bool, desc: str = ""):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    torch.set_grad_enabled(train)
    pbar = tqdm(loader, desc=desc, leave=False)
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_batches += labels.size(0)
        all_preds.append(logits.detach().argmax(dim=-1).cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.set_grad_enabled(True)
    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, average="macro")
    return avg_loss, acc, f1


def train_model(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    trainable_state_dict_fn,
    weight_decay: float = 0.01,
) -> RunResult:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ckpt_mb = state_dict_size_mb(trainable_state_dict_fn(model))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    result = RunResult(name=name, trainable_params=trainable, total_params=total, checkpoint_size_mb=ckpt_mb)

    with GPUMemoryTracker(device) as mem:
        run_start = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, _, _ = run_one_epoch(
                model, train_loader, optimizer, device, criterion, train=True,
                desc=f"[{name}] epoch {epoch}/{epochs} train",
            )
            val_loss, val_acc, val_f1 = run_one_epoch(
                model, val_loader, optimizer, device, criterion, train=False,
                desc=f"[{name}] epoch {epoch}/{epochs} val",
            )
            dt = time.time() - t0
            result.epochs.append(EpochLog(epoch, train_loss, val_loss, val_acc, val_f1, dt))
            print(
                f"[{name}] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({dt:.1f}s)"
            )
        result.total_train_seconds = time.time() - run_start

    result.peak_gpu_mem_mb = mem.peak_mb
    return result
