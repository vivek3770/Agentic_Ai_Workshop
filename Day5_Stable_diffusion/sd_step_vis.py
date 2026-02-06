# sd_step_vis.py
import argparse, os, math, time
import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def decode_latents_to_pil(pipe, latents: torch.Tensor):
    """
    Decode latents -> PIL (B,H,W,3), handling SD 1.x scaling.
    latents: [B,4,H/8,W/8]
    """
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor  # ~0.18215
        imgs = pipe.vae.decode(latents, return_dict=False)[0]
        imgs = (imgs / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy() # permute reorders tensor dimensions. Here’s why it’s needed and what it does:The Problem: Different Tensor Layout ConventionsPyTorch/neural networks typically use: [Batch, Channels, Height, Width] (BCHW)PIL/NumPy images expect: [Batch, Height, Width, Channels] (BHWC)
        imgs = (imgs * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in imgs]

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id")
    ap.add_argument("--prompt", required=True, help="Positive prompt")
    ap.add_argument("--negative", default="", help="Negative prompt (for CFG)")
    ap.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    ap.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale (>=1)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="sd_steps", help="Folder to save per-step images")
    args = ap.parse_args(args)

    os.makedirs(args.outdir, exist_ok=True)
    device = pick_device()
    print(f"Device: {device.type}")

    # Load pipeline
    # NOTE: float16 works on mps; on CPU we keep float32
    dtype = torch.float16 if device.type == "mps" else torch.float32 #dtype is the data type of tensors in PyTorch.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None, # simplify demo / avoid extra overhead
    ) #Loads the pipeline from Hugging Face, Sets the dtype., Disables the safety checker to reduce overhead.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) # Replaces the default scheduler with DPMSolverMultistepScheduler (fewer steps).
    pipe = pipe.to(device)

    # Small memory tweaks
    pipe.enable_attention_slicing() #Enables attention slicing to reduce memory usage.

    # Seeded RNG for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # Prepare text embeddings (cond/uncond) once
    text_inputs = pipe.tokenizer(
        [args.prompt], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    neg_inputs = pipe.tokenizer(
        [args.negative], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        cond_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_embeds = pipe.text_encoder(neg_inputs.input_ids.to(device))[0]
        text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)  # [2, seq, dim]

    # 1) Initialize random latent z_T
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8),
        generator=generator, device=device, dtype=dtype
    )

    # 2) Set the scheduler steps/time grid
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 3) Denoise loop with CFG
    t0 = time.time()
    for i, t in enumerate(timesteps):
        # Expand latents for CFG: duplicate batch (uncond, cond)
        latent_model_input = torch.cat([latents, latents], dim=0)
        # Scale by scheduler (diffusers convention)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep=t)

        # U-Net forward (twice in batch): predicts noise
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

        # Split uncond/cond and combine via CFG
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise = noise_uncond + args.guidance * (noise_cond - noise_uncond)

        # Scheduler step: z_{t-1}
        latents = pipe.scheduler.step(noise, t, latents).prev_sample

        # 4) Decode and save preview for this step
        step_imgs = decode_latents_to_pil(pipe, latents)
        step_path = os.path.join(args.outdir, f"step_{i+1:03d}.png")
        step_imgs[0].save(step_path)
        print(f"[{i+1:02d}/{args.steps}] saved {step_path}")

    print(f"Done in {time.time()-t0:.1f}s. Final image saved as last step.")

    # Optional: also save a final copy
    final_path = os.path.join(args.outdir, "final.png")
    decode_latents_to_pil(pipe, latents)[0].save(final_path)
    print(f"Final image: {final_path}")

if __name__ == "__main__":
    # Example of how to call main with arguments in a notebook
    # Replace 'a photo of an astronaut riding a horse on mars' with your desired prompt
    main(args=["--prompt", "a photo of an astronaut riding a horse on mars", "--steps", "5"])
