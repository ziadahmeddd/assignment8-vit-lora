"""
Generates notebooks/assignment8_kaggle.ipynb AND notebooks/assignment8_colab.ipynb
— single, fully self-contained notebooks (no imports from this repo's src/,
since neither host puts this repo on PYTHONPATH by default). Both notebooks
share the exact same training code; only the handful of platform-specific
bits (GPU-setup instructions, package top-up, output directory) differ.

Run once locally:

    .venv/Scripts/python notebooks/build_notebook.py

then either:
  - upload assignment8_kaggle.ipynb to Kaggle (File > Upload Notebook), turn
    on GPU + Internet in the notebook's settings panel, and Run All; or
  - upload assignment8_colab.ipynb to Google Colab (File > Upload notebook,
    or open it from Drive/GitHub), set Runtime > Change runtime type >
    T4 GPU, and Run All.
"""
import nbformat as nbf


def build_notebook(target: str):
    assert target in ("kaggle", "colab")
    nb = nbf.v4.new_notebook()
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    if target == "kaggle":
        platform_label = "Kaggle"
        setup_note = (
            "**Before running:** in the notebook's right-hand *Settings* panel, set "
            "**Accelerator = GPU** and **Internet = On** (needed to download the "
            "dataset from Zenodo and the pretrained weights from the Hub)."
        )
        work_dir_expr = 'Path("/kaggle/working")'
        work_dir_comment = ""
        override_comment = (
            "# If you instead attached a Kaggle dataset with EuroSAT already extracted,\n"
            '# point this at it directly, e.g. Path("/kaggle/input/eurosat-rgb/EuroSAT_RGB")'
        )
        override_example = 'Path("/kaggle/input/<slug>/EuroSAT_RGB")'
    else:
        platform_label = "Colab"
        setup_note = (
            "**Before running:** in the menu bar choose **Runtime -> Change runtime "
            "type**, set **Hardware accelerator = T4 GPU** (or better), and Save. "
            "Colab has internet access on by default."
        )
        work_dir_expr = 'Path("/content")'
        work_dir_comment = (
            "\n# Colab's local disk is ephemeral (wiped when the runtime disconnects).\n"
            "# To persist results across sessions, mount Drive first and point WORK_DIR\n"
            "# there instead, e.g.:\n"
            "#   from google.colab import drive; drive.mount('/content/drive')\n"
            "#   WORK_DIR = Path('/content/drive/MyDrive/assignment8_outputs')"
        )
        override_comment = (
            "# If you already have EuroSAT extracted somewhere (e.g. a Drive folder\n"
            "# you mounted above), point this at it directly to skip the download."
        )
        override_example = "Path('/content/drive/MyDrive/EuroSAT_RGB')"

    # -----------------------------------------------------------------------
    md(rf"""
# Assignment 8 — Full Fine-Tuning vs. Manual LoRA on Vision Transformer ({platform_label})

Adapts `google/vit-base-patch16-224` (`ViTForImageClassification`) to the
**EuroSAT RGB** land-use classification dataset (10 classes, 27,000 images),
comparing:

1. **Full fine-tuning** — every parameter trainable.
2. **Manual LoRA** — implemented from scratch (no PEFT library anywhere in
   this notebook), applied to the Query and Value projections of the
   **last three transformer blocks**, swept over rank ∈ {{16, 32, 64}}.
3. A bonus sweep over **which layers** receive LoRA (objectives list also
   calls out "different target layers"), at the best rank from step 2.

For every run we record: trainable params, train/val loss, accuracy,
macro-F1, peak GPU memory, training time, and adapter/checkpoint size.

{setup_note}
""")

    # -----------------------------------------------------------------------
    code(r"""
import subprocess, sys
def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# Tops up whatever the host notebook doesn't already ship, without touching
# (and possibly downgrading) the preinstalled torch build.
try:
    import sklearn  # noqa
except ImportError:
    pip_install("scikit-learn")

try:
    import transformers  # noqa
except ImportError:
    pip_install("transformers")

import os, io, math, time, zipfile, random, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from transformers import ViTForImageClassification, ViTImageProcessor

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
""")

    # -----------------------------------------------------------------------
    code(rf"""
# ----------------------------- Config ---------------------------------------
MODEL_NAME = "google/vit-base-patch16-224"
SEED = 42
VAL_FRACTION = 0.2
BATCH_SIZE = 32
EPOCHS = 6                 # same epoch budget for every run -> fair comparison
LR_FULL_FINETUNE = 5e-5
LR_LORA = 1e-3
LORA_RANKS = [16, 32, 64]
LORA_TARGET_MODULES = ("query", "value")
LORA_TARGET_BLOCKS = (-3, -2, -1)          # last three transformer blocks
RUN_TARGET_LAYER_BONUS_SWEEP = True         # objectives list: "different target layers"

WORK_DIR = {work_dir_expr}{work_dir_comment}
DATA_DIR = WORK_DIR / "data"
EUROSAT_ZENODO_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
{override_comment}
EUROSAT_ROOT_OVERRIDE = None  # e.g. {override_example}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
NUM_LABELS = len(EUROSAT_CLASSES)
id2label = {{i: c for i, c in enumerate(EUROSAT_CLASSES)}}
label2id = {{c: i for i, c in id2label.items()}}
""")

    # -----------------------------------------------------------------------
    md("## Data — download EuroSAT_RGB.zip (Zenodo record 7711810) and build a stratified train/val split")

    code(r"""
def download_and_extract_eurosat(dest_dir: Path, url: str = EUROSAT_ZENODO_URL) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / "EuroSAT_RGB"
    if extracted.exists() and any(extracted.iterdir()):
        return extracted
    zip_path = dest_dir / "EuroSAT_RGB.zip"
    if not zip_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    if not extracted.exists():
        for candidate in dest_dir.rglob("AnnualCrop"):
            extracted = candidate.parent
            break
    return extracted


if EUROSAT_ROOT_OVERRIDE is not None:
    EUROSAT_ROOT = EUROSAT_ROOT_OVERRIDE
else:
    EUROSAT_ROOT = download_and_extract_eurosat(DATA_DIR)
print("EuroSAT root:", EUROSAT_ROOT)
assert (EUROSAT_ROOT / "AnnualCrop").is_dir(), "Expected class folders under EUROSAT_ROOT"
""")

    code(r"""
def list_samples(root: Path) -> List[Tuple[str, int]]:
    samples = []
    for class_idx, class_name in enumerate(EUROSAT_CLASSES):
        class_dir = root / class_name
        for img_path in sorted(class_dir.glob("*.jpg")):
            samples.append((str(img_path), class_idx))
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def stratified_split(samples, val_fraction=VAL_FRACTION, seed=SEED):
    rng = random.Random(seed)
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))
    train, val = [], []
    for label, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


all_samples = list_samples(EUROSAT_ROOT)
train_samples, val_samples = stratified_split(all_samples)
print(f"total={len(all_samples)}  train={len(train_samples)}  val={len(val_samples)}")
""")

    code(r"""
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
img_size = processor.size.get("height", processor.size.get("shortest_edge", 224))
mean, std = processor.image_mean, processor.image_std

from torchvision import transforms as T

train_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),          # satellite imagery has no canonical "up"
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])
eval_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])


class EuroSATDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


train_ds = EuroSATDataset(train_samples, train_transform)
val_ds = EuroSATDataset(val_samples, eval_transform)

NUM_WORKERS = 2
# persistent_workers keeps one worker pool alive for the whole run instead of
# respawning it every single epoch (train + val) x (8 experiments) — that respawn
# churn is what produces "can only test a child process" __del__ spam on some
# hosted Jupyter kernels (observed on Kaggle). Harmless either way, but this
# also removes the noise and is a bit faster.
_loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **_loader_kwargs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **_loader_kwargs)
print(f"train batches={len(train_loader)}  val batches={len(val_loader)}")
""")

    # -----------------------------------------------------------------------
    md(r"""
## Manual LoRA implementation (from scratch — no PEFT library)

`LoRALinear` wraps an existing `nn.Linear`, freezes its weight/bias, and adds
a trainable low-rank update:

$$h = W_0 x + b_0 + \frac{\alpha}{r} B (A x)$$

with $A \in \mathbb{R}^{r \times d_{in}}$ Kaiming-initialized and
$B \in \mathbb{R}^{d_{out} \times r}$ initialized to **zero**, so the adapter
starts as an exact no-op and training only ever *adds* a correction on top of
the frozen pretrained weights.
""")

    code(r"""
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = None, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = None
    dropout: float = 0.0
    target_modules: tuple = ("query", "value")
    target_blocks: tuple = None  # None => all blocks


def _get_encoder_layers(model):
    vit = model.vit
    if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        return vit.encoder.layer          # transformers < 5
    if hasattr(vit, "layers"):
        return vit.layers                 # transformers >= 5
    raise AttributeError("Could not locate ViT transformer blocks on this model.")


_TARGET_CANDIDATES = {
    "query": [("attention.attention", "query"), ("attention", "q_proj")],
    "key": [("attention.attention", "key"), ("attention", "k_proj")],
    "value": [("attention.attention", "value"), ("attention", "v_proj")],
    "output": [("attention.output", "dense"), ("attention", "o_proj")],
    "intermediate": [("intermediate", "dense"), ("mlp", "fc1")],
    "mlp_output": [("output", "dense"), ("mlp", "fc2")],
}


def _resolve_parent(block, dotted_path):
    obj = block
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _find_target(block, name):
    for parent_path, attr in _TARGET_CANDIDATES[name]:
        try:
            parent = _resolve_parent(block, parent_path)
        except AttributeError:
            continue
        if hasattr(parent, attr) and isinstance(getattr(parent, attr), nn.Linear):
            return parent, attr
    raise AttributeError(f"Could not locate target module '{name}'")


def _resolve_target_blocks(num_layers, target_blocks):
    if target_blocks is None:
        return list(range(num_layers))
    resolved = []
    for b in target_blocks:
        idx = b if b >= 0 else num_layers + b
        assert 0 <= idx < num_layers, f"block {b} out of range"
        resolved.append(idx)
    return resolved


def apply_lora_to_vit(model, config: LoRAConfig) -> List[str]:
    encoder_layers = _get_encoder_layers(model)
    num_layers = len(encoder_layers)
    block_indices = _resolve_target_blocks(num_layers, config.target_blocks)
    replaced = []
    for layer_idx in block_indices:
        block = encoder_layers[layer_idx]
        for name in config.target_modules:
            parent, attr = _find_target(block, name)
            original = getattr(parent, attr)
            wrapped = LoRALinear(original, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
            setattr(parent, attr, wrapped)
            replaced.append(f"vit.layer.{layer_idx}.{name}")
    return replaced


def freeze_all_but_lora_and_head(model, head_attr="classifier"):
    for name, param in model.named_parameters():
        is_lora = ".lora_A" in name or ".lora_B" in name
        is_head = name.startswith(f"{head_attr}.")
        param.requires_grad = is_lora or is_head


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_state_dict(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.cpu() for k, v in model.state_dict().items() if k in names}
""")

    # -----------------------------------------------------------------------
    md("## Training + metrics harness (shared by every run)")

    code(r"""
def state_dict_size_mb(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


class GPUMemoryTracker:
    def __init__(self, device):
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
    epochs: list = field(default_factory=list)
    total_train_seconds: float = 0.0
    peak_gpu_mem_mb: float = 0.0

    @property
    def final(self):
        return self.epochs[-1]

    def as_row(self):
        e = self.final
        return {
            "run": self.name,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_%": round(100 * self.trainable_params / self.total_params, 4),
            "checkpoint_MB": round(self.checkpoint_size_mb, 2),
            "train_loss": round(e.train_loss, 4),
            "val_loss": round(e.val_loss, 4),
            "val_accuracy": round(e.val_accuracy, 4),
            "val_f1_macro": round(e.val_f1_macro, 4),
            "train_seconds": round(self.total_train_seconds, 1),
            "peak_gpu_mem_MB": round(self.peak_gpu_mem_mb, 1),
        }


def run_one_epoch(model, loader, optimizer, device, criterion, train: bool, desc: str = ""):
    model.train(mode=train)
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    torch.set_grad_enabled(train)
    pbar = tqdm(loader, desc=desc, leave=False)
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_preds.append(outputs.logits.detach().argmax(dim=-1).cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    torch.set_grad_enabled(True)
    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return (total_loss / max(n, 1),
            accuracy_score(labels_np, preds),
            f1_score(labels_np, preds, average="macro"))


def train_model(name, model, train_loader, val_loader, device, epochs, lr, trainable_state_dict_fn, weight_decay=0.01):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ckpt_mb = state_dict_size_mb(trainable_state_dict_fn(model))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    result = RunResult(name=name, trainable_params=trainable, total_params=total, checkpoint_size_mb=ckpt_mb)

    with GPUMemoryTracker(device) as mem:
        t_start = time.time()
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
            print(f"[{name}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({dt:.1f}s)")
        result.total_train_seconds = time.time() - t_start
    result.peak_gpu_mem_mb = mem.peak_mb
    return result
""")

    # -----------------------------------------------------------------------
    md("## Experiment 1 — Full fine-tuning (every parameter trainable)")

    code(r"""
full_model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
).to(device)
for p in full_model.parameters():
    p.requires_grad = True

full_result = train_model(
    "full_finetune", full_model, train_loader, val_loader, device,
    epochs=EPOCHS, lr=LR_FULL_FINETUNE,
    trainable_state_dict_fn=lambda m: {k: v.cpu() for k, v in m.state_dict().items()},
)

del full_model
if device.type == "cuda":
    torch.cuda.empty_cache()
full_result.as_row()
""")

    # -----------------------------------------------------------------------
    md(r"""
## Experiment 2 — Manual LoRA on Query & Value of the last three transformer blocks
### Rank sweep: r ∈ {16, 32, 64}
""")

    code(r"""
def build_lora_model(lora_cfg: LoRAConfig):
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )
    replaced = apply_lora_to_vit(model, lora_cfg)
    freeze_all_but_lora_and_head(model)
    return model, replaced


lora_rank_results = []
for rank in LORA_RANKS:
    cfg = LoRAConfig(rank=rank, target_modules=LORA_TARGET_MODULES, target_blocks=LORA_TARGET_BLOCKS)
    model, replaced = build_lora_model(cfg)
    model.to(device)
    print(f"[lora_r{rank}] LoRA applied to: {replaced}")
    result = train_model(
        f"lora_r{rank}_QV_last3blocks", model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LR_LORA, trainable_state_dict_fn=trainable_state_dict,
    )
    lora_rank_results.append(result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

pd.DataFrame([r.as_row() for r in lora_rank_results])
""")

    # -----------------------------------------------------------------------
    md(r"""
## Bonus — Different target layers (objectives list explicitly asks to study this)

Fixing rank = the best-performing rank from the sweep above, compare which
projections receive LoRA: Query+Value only, Query+Key+Value, +output
projection, and +MLP layers — all still restricted to the last three blocks.
""")

    code(r"""
best_rank_result = max(lora_rank_results, key=lambda r: r.final.val_accuracy)
best_rank = int(best_rank_result.name.split("_r")[1].split("_")[0])
print("Best rank by val accuracy:", best_rank)

target_layer_results = []
if RUN_TARGET_LAYER_BONUS_SWEEP:
    variants = {
        "QV": ("query", "value"),
        "QKV": ("query", "key", "value"),
        "QKVO": ("query", "key", "value", "output"),
        "QV+MLP": ("query", "value", "intermediate", "mlp_output"),
    }
    for variant_name, modules in variants.items():
        if variant_name == "QV":
            target_layer_results.append(best_rank_result)  # already have this run
            continue
        cfg = LoRAConfig(rank=best_rank, target_modules=modules, target_blocks=LORA_TARGET_BLOCKS)
        model, replaced = build_lora_model(cfg)
        model.to(device)
        print(f"[lora_r{best_rank}_{variant_name}] LoRA applied to: {replaced}")
        result = train_model(
            f"lora_r{best_rank}_{variant_name}_last3blocks", model, train_loader, val_loader, device,
            epochs=EPOCHS, lr=LR_LORA, trainable_state_dict_fn=trainable_state_dict,
        )
        target_layer_results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

pd.DataFrame([r.as_row() for r in target_layer_results])
""")

    # -----------------------------------------------------------------------
    md("## Results comparison — full fine-tuning vs. every LoRA configuration")

    code(r"""
all_results = [full_result] + lora_rank_results + [r for r in target_layer_results if r not in lora_rank_results]
comparison_df = pd.DataFrame([r.as_row() for r in all_results])
comparison_df
""")

    code(r"""
comparison_df.to_csv(WORK_DIR / "results_comparison.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

axes[0, 0].bar(comparison_df["run"], comparison_df["trainable_params"])
axes[0, 0].set_title("Trainable parameters")
axes[0, 0].tick_params(axis="x", rotation=75)
axes[0, 0].set_yscale("log")

axes[0, 1].bar(comparison_df["run"], comparison_df["val_accuracy"])
axes[0, 1].set_title("Validation accuracy")
axes[0, 1].tick_params(axis="x", rotation=75)

axes[0, 2].bar(comparison_df["run"], comparison_df["val_f1_macro"])
axes[0, 2].set_title("Validation macro-F1")
axes[0, 2].tick_params(axis="x", rotation=75)

axes[1, 0].bar(comparison_df["run"], comparison_df["checkpoint_MB"])
axes[1, 0].set_title("Checkpoint size (MB)")
axes[1, 0].tick_params(axis="x", rotation=75)
axes[1, 0].set_yscale("log")

axes[1, 1].bar(comparison_df["run"], comparison_df["peak_gpu_mem_MB"])
axes[1, 1].set_title("Peak GPU memory (MB)")
axes[1, 1].tick_params(axis="x", rotation=75)

axes[1, 2].bar(comparison_df["run"], comparison_df["train_seconds"])
axes[1, 2].set_title("Total training time (s)")
axes[1, 2].tick_params(axis="x", rotation=75)

plt.tight_layout()
plt.savefig(WORK_DIR / "results_comparison.png", dpi=150)
plt.show()
""")

    code(r"""
plt.figure(figsize=(8, 5))
for r in all_results:
    epochs_x = [e.epoch for e in r.epochs]
    val_loss_y = [e.val_loss for e in r.epochs]
    plt.plot(epochs_x, val_loss_y, marker="o", label=r.name)
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.title("Validation loss curves")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(WORK_DIR / "val_loss_curves.png", dpi=150)
plt.show()
""")

    if target == "colab":
        md(
            "## Download results\n\n"
            "Colab's local disk is ephemeral. If you didn't mount Drive above, "
            "grab the output files now before the runtime disconnects."
        )
        code(r"""
from google.colab import files
for fname in ("results_comparison.csv", "results_comparison.png", "val_loss_curves.png"):
    files.download(str(WORK_DIR / fname))
""")

    md(r"""
## Conclusions

Fill in after the run completes, referencing `comparison_df` above:

- **Trainable parameters**: LoRA trains only ~a few % of full fine-tuning's
  parameter count (see `trainable_%` column) — increasing with rank.
- **Accuracy / F1**: compare `val_accuracy` / `val_f1_macro` across
  `full_finetune` vs. `lora_r16/32/64` — note whether higher rank keeps
  closing the gap to full fine-tuning, and where it plateaus.
- **GPU memory & training time**: LoRA's savings come from not needing
  optimizer state / gradients for frozen weights — compare `peak_gpu_mem_MB`
  and `train_seconds`.
- **Model/checkpoint size**: compare `checkpoint_MB` — this is the practical
  deployment win of LoRA (ship a tiny adapter instead of the full model).
- **Target layers**: from the bonus sweep, note whether adding K, the output
  projection, or the MLP layers to the LoRA set meaningfully changes accuracy
  for a fixed rank, and at what parameter-count cost.
""")

    # -----------------------------------------------------------------------
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    if target == "colab":
        nb["metadata"]["colab"] = {"name": "assignment8_colab.ipynb", "provenance": []}
        nb["metadata"]["accelerator"] = "GPU"

    out_path = f"notebooks/assignment8_{target}.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", out_path)


if __name__ == "__main__":
    for t in ("kaggle", "colab"):
        build_notebook(t)
