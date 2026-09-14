Assignment 8: Full Fine-Tuning vs. Manual LoRA on Vision Transformer

## Objective

In this assignment, you will adapt a pretrained Vision Transformer (ViT) to a satellite-image classification problem using the EuroSAT RGB dataset.

You will implement and compare two approaches:

## Full Fine-Tuning And LoRA Fine Tuning

LoRA Fine-Tuning implemented manually from scratch

You are not allowed to use the PEFT library's LoRA implementation for the LoRA part.

The goal is not only to achieve good accuracy, but also to understand the effect of LoRA on:

- Number of trainable parameters

- Training loss

- Validation loss

- Accuracy

- F1-score

- GPU memory

- Training time

- Model size

- Different LoRA ranks

- Different target layers

## Dataset

Use the EuroSAT RGB dataset:

EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification

## Dataset:

## [https://zenodo.org/records/7711810](https://zenodo.org/records/7711810)

EuroSAT contains 27,000 labeled satellite images belonging to 10 land-use/land-cover classes. The RGB version contains the Red, Green, and Blue bands as JPEG images.

Dataset classes


Your model must classify images into:

- AnnualCrop

Forest HerbaceousVegetation Highway Industrial Pasture PermanentCrop Residential River SeaLake

Use EuroSAT_RGB.zip.

Do not use the multispectral version for this assignment.

## Pretrained Model

Use:

google/vit-base-patch16-224

You should use:

ViTForImageClassification

Which Layers Should Receive LoRA?

For the first LoRA experiment, apply LoRA to the Query and Value projections of the last three Transformer blocks.

Experiment — LoRA Rank 16, 32, 64
