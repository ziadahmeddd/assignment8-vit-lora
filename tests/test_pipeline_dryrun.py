"""
End-to-end dry run of the training/metrics harness (metrics.train_model,
RunResult, GPUMemoryTracker, checkpoint sizing) using a tiny randomly-initialized
ViT and synthetic data — no real dataset download, no GPU, finishes in seconds.
This is purely plumbing validation; real numbers come from the Kaggle run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTConfig, ViTForImageClassification

from lora import LoRAConfig, apply_lora_to_vit, freeze_all_but_lora_and_head, trainable_state_dict
from metrics import train_model


def make_tiny_vit(num_labels=10, num_hidden_layers=4):
    cfg = ViTConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=32,
        patch_size=16,
        num_labels=num_labels,
    )
    return ViTForImageClassification(cfg)


def make_loaders(num_labels=10, n_train=32, n_val=16, batch_size=8):
    x_train = torch.randn(n_train, 3, 32, 32)
    y_train = torch.randint(0, num_labels, (n_train,))
    x_val = torch.randn(n_val, 3, 32, 32)
    y_val = torch.randint(0, num_labels, (n_val,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


def test_full_finetune_dryrun():
    device = torch.device("cpu")
    model = make_tiny_vit().to(device)
    train_loader, val_loader = make_loaders()
    result = train_model(
        name="full_finetune_dryrun",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=2,
        lr=5e-5,
        trainable_state_dict_fn=lambda m: {k: v.cpu() for k, v in m.state_dict().items()},
    )
    assert len(result.epochs) == 2
    assert result.trainable_params == result.total_params
    row = result.as_row()
    print(row)
    assert row["trainable_%"] == 100.0


def test_lora_dryrun_rank_sweep_smaller_checkpoint_than_full():
    device = torch.device("cpu")
    train_loader, val_loader = make_loaders()

    full_model = make_tiny_vit().to(device)
    full_result = train_model(
        "full", full_model, train_loader, val_loader, device, epochs=1, lr=5e-5,
        trainable_state_dict_fn=lambda m: {k: v.cpu() for k, v in m.state_dict().items()},
    )

    rows = [full_result.as_row()]
    prev_trainable = None
    for rank in (2, 4, 8):
        model = make_tiny_vit().to(device)
        cfg = LoRAConfig(rank=rank, target_modules=("query", "value"), target_blocks=(-3, -2, -1))
        apply_lora_to_vit(model, cfg)
        freeze_all_but_lora_and_head(model)
        result = train_model(
            f"lora_r{rank}", model, train_loader, val_loader, device, epochs=1, lr=1e-3,
            trainable_state_dict_fn=trainable_state_dict,
        )
        rows.append(result.as_row())
        assert result.trainable_params < full_result.trainable_params
        assert result.checkpoint_size_mb < full_result.checkpoint_size_mb
        if prev_trainable is not None:
            assert result.trainable_params > prev_trainable, "higher rank should mean more trainable params"
        prev_trainable = result.trainable_params

    print("\nComparison table:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    test_full_finetune_dryrun()
    test_lora_dryrun_rank_sweep_smaller_checkpoint_than_full()
    print("\nALL DRY-RUN PIPELINE TESTS PASSED")
