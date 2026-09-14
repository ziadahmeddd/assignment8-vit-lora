# Assignment 8 — Full Fine-Tuning vs. Manual LoRA on Vision Transformer

Adapts `google/vit-base-patch16-224` (`ViTForImageClassification`) to the
**EuroSAT RGB** dataset (10 land-use classes, 27,000 images), comparing full
fine-tuning against a **from-scratch LoRA implementation** (no PEFT library
used anywhere).

## Layout

```
Assignment 8.md              the assignment brief
src/
  lora.py                    manual LoRA (LoRALinear, apply_lora_to_vit, freezing, param counting)
  data.py                    EuroSAT download/split/Dataset
  metrics.py                 training loop, RunResult, GPU-memory/size/timing metrics
  experiments.py             orchestrates full-FT + LoRA rank sweep + target-layer sweep
tests/
  test_lora_smoke.py         LoRA correctness (no-op at init, gradient routing, param counts) — CPU, seconds
  test_pipeline_dryrun.py    end-to-end harness check with synthetic data — CPU, seconds
notebooks/
  build_notebook.py          generates the notebooks below (edit this, not the .ipynb files, then re-run it)
  assignment8_kaggle.ipynb   <-- run this on Kaggle (needs a GPU)
  assignment8_colab.ipynb    <-- run this on Google Colab (needs a GPU)
requirements.txt             pinned local venv packages (CPU-only, for dev/testing)
.venv/                       local Python 3.14 virtualenv
```

## Why a notebook for the actual training

Full fine-tuning of ViT-base plus a LoRA rank sweep (16/32/64) and a
target-layer sweep is a multi-hour job on a 27k-image dataset. That was run
on **Kaggle** (free T4/P100 GPU) rather than the local machine (6GB laptop
GPU). Everything in `src/` was validated locally first:

- `tests/test_lora_smoke.py` — verifies LoRA is a mathematical no-op at init
  (B initialized to zero), that only LoRA A/B + the classifier head receive
  gradients (base ViT weights stay frozen), and that trainable-parameter
  count scales with rank as expected.
- `tests/test_pipeline_dryrun.py` — runs the actual training loop
  (`metrics.train_model`) on synthetic tensors to check the harness end to
  end without needing the dataset or a GPU.
- Beyond that, the **real** EuroSAT_RGB.zip was downloaded and the **real**
  pretrained `google/vit-base-patch16-224` was loaded locally on CPU to run
  a 1-epoch/small-subset dry run of the exact code that ships in the
  notebook (full fine-tune → LoRA rank sweep → target-layer sweep →
  comparison table/plots), confirming it runs without errors before handing
  the heavy version to Kaggle.

Run the smoke tests yourself:

```
.venv/Scripts/python tests/test_lora_smoke.py
.venv/Scripts/python tests/test_pipeline_dryrun.py
```

## Running the real training on Kaggle

1. Upload `notebooks/assignment8_kaggle.ipynb` to Kaggle (File → Upload
   Notebook or paste its cells into a new notebook).
2. In the notebook's Settings panel: **Accelerator = GPU** (T4 x1 is
   plenty), **Internet = On** (needed to pull the dataset from Zenodo and
   the pretrained weights from the Hub).
3. Run All. It will:
   - Download & extract `EuroSAT_RGB.zip` from Zenodo record 7711810 (the
     RGB version only, as required — not multispectral).
   - Build an 80/20 per-class stratified train/val split.
   - **Experiment 1**: full fine-tuning (every parameter trainable).
   - **Experiment 2**: manual LoRA on Query & Value of the last three
     transformer blocks, rank ∈ {16, 32, 64} — exactly as specified in the
     assignment.
   - **Bonus**: at the best rank found, sweep which layers get LoRA (QV vs
     QKV vs QKVO vs QV+MLP) — the objectives list explicitly calls out
     understanding the effect of "different target layers", though the
     assignment brief's own text cuts off before detailing this experiment.
   - Save `results_comparison.csv`, `results_comparison.png`, and
     `val_loss_curves.png` to `/kaggle/working/`, and print a comparison
     table covering trainable params, train/val loss, accuracy, macro-F1,
     peak GPU memory, training time, and checkpoint size for every run.
4. To skip the Zenodo download and instead use a Kaggle Dataset you've
   attached, set `EUROSAT_KAGGLE_INPUT_OVERRIDE` in the config cell to its
   `/kaggle/input/...` path.
5. `EPOCHS`, learning rates, and `LORA_RANKS` are all single constants near
   the top of the config cell if you want to adjust the time/accuracy
   trade-off.

## Running the real training on Google Colab

Same code, packaged as `notebooks/assignment8_colab.ipynb` — Kaggle and Colab
both generate from the same `build_notebook.py`, so the only differences are
the GPU-setup step, the output directory (`/content` vs `/kaggle/working`),
and a final "download results" cell.

1. Open [colab.research.google.com](https://colab.research.google.com), then
   File → Upload notebook and pick `notebooks/assignment8_colab.ipynb` (or
   upload it to Drive/GitHub first and open it from there).
2. **Runtime → Change runtime type → Hardware accelerator = T4 GPU** (or a
   better GPU if you have Colab Pro), then Save.
3. Run All. It downloads EuroSAT from Zenodo, runs the same full-FT + LoRA
   rank sweep + target-layer bonus sweep as the Kaggle version, and writes
   `results_comparison.csv`, `results_comparison.png`, and
   `val_loss_curves.png` to `/content/`.
4. Colab's local disk is wiped when the runtime disconnects. Either run the
   last "Download results" cell to save the three output files locally, or
   mount Google Drive first and point `WORK_DIR` at a Drive folder (see the
   commented-out snippet in the config cell) so outputs persist automatically.
5. To skip the Zenodo download (e.g. you already extracted EuroSAT into a
   mounted Drive folder), set `EUROSAT_ROOT_OVERRIDE` in the config cell to
   that path.

## Notes on the manual LoRA implementation

`LoRALinear` (in `src/lora.py`, and re-embedded self-contained in the
notebook) wraps an existing `nn.Linear`:

```
h = W0 x + b0 + (alpha / r) * B(A x)
```

`W0`/`b0` are frozen; only `A` (Kaiming-init) and `B` (zero-init, so LoRA
starts as an exact no-op) are trained, alongside the newly-initialized
classification head. `apply_lora_to_vit` locates the target projections
across both the pre- and post-refactor `transformers` module layouts
(`vit.encoder.layer[i].attention.attention.{query,value}` vs.
`vit.layers[i].attention.{q_proj,v_proj}`), so it works regardless of which
`transformers` version Kaggle has pinned.
