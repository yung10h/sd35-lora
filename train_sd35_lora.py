import argparse
import csv
import os
import random
import time
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import bitsandbytes as bnb
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision import transforms
from datasets import Dataset, Features, Value
from diffusers import StableDiffusion3Pipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model, PeftType
from accelerate import Accelerator
from accelerate.state import PartialState

# Logging setup
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
PartialState()

# Argument parsing and configuration
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="image_caption_dataset")
    parser.add_argument("--output_dir", default="SD35_lora_output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--caption_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Resize and crop image
def resize_and_crop(image, size):
    w, h = image.size
    m = min(w, h)
    return image.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((size, size), Image.LANCZOS)

# Load image-caption pairs and preprocess
def load_dataset(root, resolution, tokenizers, dropout_prob):
    tokenizer1, tokenizer2, tokenizer3 = tokenizers
    caption_file = Path(root) / "captions.tsv"
    image_dir = Path(root) / "images"

    rows = []
    with open(caption_file, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for filename, caption in reader:
            image_path = image_dir / filename
            if image_path.exists():
                rows.append({"image_path": str(image_path), "text": caption})

    dataset = Dataset.from_list(rows, features=Features({
        "image_path": Value("string"),
        "text": Value("string")
    }))

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: resize_and_crop(img, resolution)),
        transforms.Lambda(lambda img: (np.asarray(img).astype(np.float16) / 127.5) - 1.0),
        transforms.Lambda(lambda arr: torch.from_numpy(arr).permute(2, 0, 1)),
    ])

    def preprocess(example):
        image = Image.open(example["image_path"]).convert("RGB")
        example["pixel_values"] = transform(image)
        if random.random() < dropout_prob:
            example["text"] = ""
        # Tokenize text with three different tokenizers
        example["input_ids_1"] = tokenizer1(example["text"], padding="max_length", max_length=tokenizer1.model_max_length, truncation=True).input_ids
        example["input_ids_2"] = tokenizer2(example["text"], padding="max_length", max_length=tokenizer2.model_max_length, truncation=True).input_ids
        example["input_ids_3"] = tokenizer3(example["text"], padding="max_length", max_length=256, truncation=True).input_ids
        return example

    dataset = dataset.map(preprocess, remove_columns=["image_path", "text"], num_proc=os.cpu_count(), desc="Preprocessing")
    return dataset

def main():
    args = parse_args()
    set_seed(args.seed)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="fp16"
    )

    tok1, tok2, tok3 = pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
    txt_enc1 = pipe.text_encoder.to(accelerator.device).eval()
    txt_enc2 = pipe.text_encoder_2.to(accelerator.device).eval()
    txt_enc3 = pipe.text_encoder_3.to(accelerator.device).eval()
    for enc in (txt_enc1, txt_enc2, txt_enc3):
        for p in enc.parameters(): p.requires_grad = False

    dataset = load_dataset(args.data_root, args.resolution, (tok1, tok2, tok3), args.caption_dropout)
    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # VAE and noise scheduler
    vae = pipe.vae.to(accelerator.device, dtype=torch.float16).eval()
    scheduler = DDPMScheduler(beta_schedule="linear", num_train_timesteps=pipe.scheduler.config.num_train_timesteps)
    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device)

    # Prepare Transformer as the denoiser and wrap it with LoRA
    denoiser = pipe.transformer
    denoiser.register_to_config(model_type="sd3_transformer2d")
    lora_config = LoraConfig(
        peft_type=PeftType.LORA,
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        init_lora_weights="gaussian"
    )
    denoiser = get_peft_model(denoiser, lora_config)
    for name, param in denoiser.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    optimizer = bnb.optim.AdamW8bit(denoiser.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.train_steps - args.warmup_steps, eta_min=args.lr / 20)
    lr_scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])

    denoiser, optimizer, dataloader, lr_scheduler, vae = accelerator.prepare(
        denoiser, optimizer, dataloader, lr_scheduler, vae
    )

    # -----  Training Loop -----
    global_step = 0
    start_time = time.time()
    pipe.set_progress_bar_config(disable=True)

    while global_step < args.train_steps:
        for batch in dataloader:
            with accelerator.accumulate(denoiser):
                imgs = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents, dtype=torch.float16)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=latents.device).long()
                noisy_latents = scheduler.add_noise(latents, noise, t)

                with torch.no_grad():
                    prompt_embed1 = txt_enc1(batch["input_ids_1"].to(accelerator.device), return_dict=True)
                    prompt_embed2 = txt_enc2(batch["input_ids_2"].to(accelerator.device), return_dict=True)
                    prompt_embed3 = txt_enc3(batch["input_ids_3"].to(accelerator.device), return_dict=True)

                enc_state1 = prompt_embed1.last_hidden_state.to(dtype=torch.float16) # (B, 77, 768)
                enc_state2 = prompt_embed2.last_hidden_state.to(dtype=torch.float16) # (B, 77, 1280)
                enc_state3 = prompt_embed3.last_hidden_state.to(dtype=torch.float16) # (B, 256, 4096)

                clip_enc_state = torch.cat([enc_state1, enc_state2], dim=-1) # (B, 77, 2048)
                clip_enc_state = nn.functional.pad( # (B, 77, 4096)
                    clip_enc_state,
                    (0, enc_state3.shape[-1] - clip_enc_state.shape[-1])
                    )

                prompt_embeds = torch.cat([clip_enc_state, enc_state3], dim=-2).to(dtype=torch.float16) # (B, 333, 4096)

                feat1 = prompt_embed1.last_hidden_state[:, 0, :].to(dtype=torch.float16) # (B, 768)
                feat2 = prompt_embed2.last_hidden_state[:, 0, :].to(dtype=torch.float16) # (B, 1280)
                pooled_prompt_embeds = torch.cat([feat1, feat2], dim=-1).to(dtype=torch.float16) # (B, 2048)

                pred = denoiser(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds, # 3D tensor for cross-attention
                    pooled_projections=pooled_prompt_embeds,
                    timestep=t
                ).sample

                # Calculate SNR-weighted loss
                snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
                weight = snr / (snr + args.snr_gamma)
                loss = (weight * (pred - noise).pow(2)).mean()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Logging
            if accelerator.is_main_process and global_step % 50 == 0:
                log.info(f"Step {global_step:5d} | Loss {loss:.4f} | LR {lr_scheduler.get_last_lr()[0]:.2e}")

            if global_step >= args.train_steps:
                break

            # Save checkpoint
            if accelerator.is_main_process and global_step % 1000 == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                denoiser.save_pretrained(checkpoint_dir)
                lora_config.save_pretrained(checkpoint_dir)
                log.info(f"Checkpoint saved → {checkpoint_dir}")
    # ----------
    
    # Save final model
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        denoiser.save_pretrained(args.output_dir)
        lora_config.save_pretrained(args.output_dir)
        log.info(f"Training completed in {(time.time() - start_time) / 60:.1f} minutes. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
