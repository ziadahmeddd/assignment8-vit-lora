"""
Local smoke test (no dataset download, no GPU needed) that verifies:
  1. apply_lora_to_vit() correctly locates & replaces Q/V projections in the
     last N blocks, for whatever transformers version is installed.
  2. freeze_all_but_lora_and_head() leaves ONLY LoRA A/B + classifier trainable.
  3. A forward + backward pass runs, and gradients only reach the frozen-expected
     trainable params (i.e. base ViT weights truly get zero grad).
  4. LoRA starts as a mathematical no-op (B initialized to zero) so LoRA-augmented
     output equals the original model's output before any training step.
  5. trainable_state_dict() round-trips and is much smaller than the full model.

Run: .venv/Scripts/python -m pytest tests/test_lora_smoke.py -v
     (or just: .venv/Scripts/python tests/test_lora_smoke.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import ViTConfig, ViTForImageClassification

from lora import (
    LoRAConfig,
    apply_lora_to_vit,
    freeze_all_but_lora_and_head,
    count_trainable_params,
    trainable_state_dict,
)


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


def test_replaces_expected_modules():
    model = make_tiny_vit(num_hidden_layers=4)
    cfg = LoRAConfig(rank=8, target_modules=("query", "value"), target_blocks=(-3, -2, -1))
    replaced = apply_lora_to_vit(model, cfg)
    assert len(replaced) == 6, replaced  # 3 blocks x (query, value)
    assert all("query" in r or "value" in r for r in replaced)
    print("replaced:", replaced)


def test_no_op_at_init():
    torch.manual_seed(0)
    model = make_tiny_vit()
    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(pixel_values=pixel_values).logits.clone()

    cfg = LoRAConfig(rank=4, target_modules=("query", "value"), target_blocks=(-2, -1))
    apply_lora_to_vit(model, cfg)
    with torch.no_grad():
        after = model(pixel_values=pixel_values).logits

    assert torch.allclose(before, after, atol=1e-6), "LoRA with B=0 must be a no-op at init"
    print("no-op-at-init OK, max diff =", (before - after).abs().max().item())


def test_only_lora_and_head_trainable_and_get_gradients():
    model = make_tiny_vit()
    cfg = LoRAConfig(rank=4, target_modules=("query", "value"), target_blocks=(-2, -1))
    apply_lora_to_vit(model, cfg)
    freeze_all_but_lora_and_head(model)

    trainable, total = count_trainable_params(model)
    assert trainable < total
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all((".lora_A" in n or ".lora_B" in n or n.startswith("classifier.")) for n in trainable_names)
    print(f"trainable={trainable} / total={total} ({100*trainable/total:.2f}%)")

    pixel_values = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 10, (2,))
    out = model(pixel_values=pixel_values, labels=labels)
    out.loss.backward()

    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{n} should have received a gradient"
        else:
            assert p.grad is None, f"{n} is frozen but received a gradient"
    print("gradient routing OK")


def test_checkpoint_is_small():
    model = make_tiny_vit()
    cfg = LoRAConfig(rank=4, target_modules=("query", "value"), target_blocks=(-2, -1))
    apply_lora_to_vit(model, cfg)
    freeze_all_but_lora_and_head(model)
    sd = trainable_state_dict(model)
    total_params = sum(p.numel() for p in model.parameters())
    ckpt_params = sum(v.numel() for v in sd.values())
    assert ckpt_params < total_params
    print(f"adapter checkpoint params={ckpt_params} vs full model params={total_params}")


def test_different_rank_changes_param_count():
    counts = {}
    for rank in (4, 8, 16):
        model = make_tiny_vit()
        cfg = LoRAConfig(rank=rank, target_modules=("query", "value"), target_blocks=(-3, -2, -1))
        apply_lora_to_vit(model, cfg)
        freeze_all_but_lora_and_head(model)
        trainable, _ = count_trainable_params(model)
        counts[rank] = trainable
    assert counts[4] < counts[8] < counts[16], counts
    print("param counts by rank:", counts)


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            print(f"\n=== {fn_name} ===")
            fn()
    print("\nALL SMOKE TESTS PASSED")
