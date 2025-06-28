# LoRA Fine-Tuning Script for Stable Diffusion 3.5 Medium

This repository provides a training script for fine-tuning **Stable Diffusion 3.5 Medium** using LoRA (Low-Rank Adaptation). IT implements a triple-tokenizer and triple-encoder architecture, enabling the use of all three text encoders from SD 3.5 Medium.
> **Note:** No model weights or LoRA adapters are included in this repository.

## Key Features

- LoRA fine-tuning of the SD 3.5 transformer module  
- Triple-tokenizer and triple-encoder integration  
- Caption dropout for robust training

## Dataset Structure

Expected layout:
```
image_caption_dataset/
├── captions.tsv
└── images/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

Example captions.tsv:
```
filename	text
image001.png	A cat sitting on a sofa.
image002.png	A bowl of ramen with sliced pork.
```

## License
This repository’s code is released under the Apache License 2.0.

## Acknowledgements
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers/)
- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
