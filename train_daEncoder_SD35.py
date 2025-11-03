import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from argparse import ArgumentParser
import warnings

from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
import diffusers
from diffusers import AutoencoderKL, StableDiffusion3Pipeline
import lpips
from safetensors import safe_open

# Debugging libs: ###############
# import matplotlib.pyplot as plt
# import numpy as np
#################################

from core.utils.common import instantiate_from_config, calculate_psnr_pt, to

import matplotlib.pyplot as plt

def compute_loss(gt, z_gt, z_pred, xhat_lq, xhat_gt, lpips_model, loss_mode, scales):
    #  "mse_ls", "ls_only", "ls_gt", "ls_gt_perceptual"
    mse_loss = 0.
    ls_loss = 0.
    gt_loss = 0.
    perceptual_loss = 0.
    loss_dict = {"mse": mse_loss, "lsa": ls_loss, "perceptual": perceptual_loss, "gt_loss": gt_loss}
    if "ls" in loss_mode:
        ls_loss = scales.lsa * F.mse_loss(z_pred, z_gt, reduction="mean")
        loss_dict["lsa"] = ls_loss.item()
        if "mse" in loss_mode:
            mse_loss = scales.mse * F.mse_loss(xhat_lq, xhat_gt, reduction="mean")
            loss_dict["mse"] = mse_loss.item()
        if "gt" in loss_mode:
            gt_loss = scales.gt * F.l1_loss(xhat_lq, xhat_gt, reduction="mean")
            loss_dict["gt_loss"] = gt_loss.item()
        if "perceptual" in loss_mode:
            perceptual_loss = scales.perceptual * lpips_model(xhat_lq, xhat_gt)
            loss_dict["perceptual"] = perceptual_loss.item()
    else:
        raise NotImplementedError("[!] Always use Latent Space Alignment (LSA) loss")

    total_loss = mse_loss + ls_loss + gt_loss + perceptual_loss
    return total_loss, loss_dict
    
#################################################################
# Call post VAE Decode: (Will be useful for stage 2 training)
#################################################################
vae_scale_factor = 1.5305
vae_shift_factor = 0.0609

def process_in(latent):
        return (latent - vae_shift_factor) * vae_scale_factor

def process_out(latent):
    return (latent / vae_scale_factor) + vae_shift_factor
#################################################################


#################################################################
# VAE
#################################################################
from core.model.vae import SDVAE

def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e
            
CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large_turbo": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
    "sd3.5_large_controlnet_blur": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_canny": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_depth": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
}
#################################################################

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(310)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    # assert cfg.train.loss_mode in ["mse_ls", "ls_only", "ls_gt", "ls_gt_perceptual"], f"Please choose a supported loss_mode from: ['mse_ls', 'ls_only', 'ls_gt', 'ls_gt_perceptual']"
    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")


    # Create & load VAE from pretrained SD 3.5 model:
    MODEL = cfg.train.sd_path
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        vae = SDVAE(device="cpu", dtype=torch.float32)
        prefix = ""
        if any(k.startswith("first_stage_model.") for k in f.keys()):
            prefix = "first_stage_model."
        load_into(f, vae, prefix, "cpu", torch.float32)


    # Make the encoder & quant_conv trainable and rest frozen
    for name, p in vae.named_parameters():
        p.requires_grad = True if "encoder" in name else False
        # print(f"{name} -> {p.shape} isTrainable? {p.requires_grad}")

    print(f"[~] All trainable VAE parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"[~] All VAE parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")

    # Setup optimizer:
    opt = torch.optim.AdamW(
        list(vae.encoder.parameters()), 
        lr=cfg.train.learning_rate, 
        weight_decay=0)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training/inference:
    vae.to(device)
    vae.encoder, opt, loader, val_loader = accelerator.prepare(
        vae.encoder, opt, loader, val_loader
    )
    vae.encoder = accelerator.unwrap_model(vae.encoder)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    step_ls_loss = []
    step_gt_loss = []
    step_perceptual_loss = []
    epoch = 0
    epoch_loss = []

    if "perceptual" in cfg.train.loss_mode:
        with warnings.catch_warnings():
            # avoid warnings from lpips internal
            warnings.simplefilter("ignore")
            lpips_model = (
                lpips.LPIPS(net="vgg", verbose=accelerator.is_local_main_process)
                .eval()
                .to(device)
            )

    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    # Training loop:
    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_local_main_process,
            unit="batch",
            total=len(loader),
        )

        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt, gt_path = batch
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # Train step:
            with torch.no_grad():
                gt_latent = vae.encode(gt).float()
                xhat_gt = vae.decode(gt_latent).clamp(-1,1).float()

            pred_latent = vae.encode(lq).float()
            # print("[+] gt latent.shape:", gt_latent.size())

            xhat_lq = vae.decode(pred_latent).clamp(-1,1).float()

            loss, loss_dict = compute_loss(gt, gt_latent, pred_latent, xhat_lq, xhat_gt,
                                           lpips_model, cfg.train.loss_mode, cfg.train.loss_scales)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            accelerator.wait_for_everyone()
          
            global_step += 1
            step_loss.append(loss_dict["mse"])
            step_ls_loss.append(loss_dict["lsa"])
            step_gt_loss.append(loss_dict["gt_loss"])
            step_perceptual_loss.append(loss_dict["perceptual"])
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 or global_step == 1:
                # Gather values from all processes
                avg_mse_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_ls_loss = (
                    accelerator.gather(
                        torch.tensor(step_ls_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_gt_loss = (
                    accelerator.gather(
                        torch.tensor(step_gt_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_perceptual_loss = (
                    accelerator.gather(
                        torch.tensor(step_perceptual_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                step_ls_loss.clear()
                step_gt_loss.clear()
                step_perceptual_loss.clear()

                if accelerator.is_local_main_process:
                    writer.add_scalar("train/mse_loss_step", avg_mse_loss, global_step)
                    writer.add_scalar("train/ls_loss_step", avg_ls_loss, global_step)
                    writer.add_scalar("train/gt_loss_step", avg_gt_loss, global_step)
                    writer.add_scalar("train/perceptual_loss_step", avg_perceptual_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0:
                if accelerator.is_local_main_process:
                    checkpoint = vae.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # Log images
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                vae.encoder.eval()
                N = 12
                log_gt, log_lq = gt[:N], lq[:N]
                with torch.no_grad():
                    log_pred = vae.decode(vae.encode(log_lq)).clamp(-1,1)
                    log_pred_gt = vae.decode(vae.encode(log_gt)).clamp(-1,1)
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/pred_gt", (log_pred_gt+1.) / 2.),
                        ("image/pred", (log_pred+1.)/2.),
                        ("image/gt", (log_gt+1.)/2.),
                        ("image/lq", (log_lq + 1.) / 2.),
                    ]:
                        writer.add_image(tag, make_grid(image, nrow=4), global_step)
                vae.encoder.train()


            # Evaluate model:
            if global_step % cfg.train.val_every == 0:
                vae.encoder.eval() 

                val_loss = []
                val_lpips_loss = []
                val_psnr = []
                val_pbar = tqdm(
                    iterable=None,
                    disable=not accelerator.is_local_main_process,
                    unit="batch",
                    total=len(val_loader),
                    leave=False,
                    desc="Validation",
                )
                for val_batch in val_loader:
                    to(val_batch, device)
                    val_batch = batch_transform(val_batch)
                    val_gt, val_lq, val_prompt, val_gt_path = val_batch
                    val_gt = (
                        rearrange(val_gt, "b h w c -> b c h w")
                        .contiguous()
                        .float()
                    ) 
                    val_lq = (
                        rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
                    )
                    with torch.no_grad():

                        lq_z = vae.encode(val_lq).float()
                        gt_z = vae.encode(val_gt).float()
                        xhat_lq = vae.decode(lq_z).clamp(0,1).float()
                        xhat_gt = vae.decode(gt_z).clamp(0,1).float()
                        vloss, vloss_dict = compute_loss(val_gt, gt_z, lq_z, xhat_lq, xhat_gt,
                                           lpips_model, cfg.train.loss_mode, cfg.train.loss_scales)

                        val_psnr.append(
                            calculate_psnr_pt(xhat_lq, val_gt, crop_border=0)
                            .mean()
                            .item()
                        )
                        val_loss.append(vloss.item())
                        val_lpips_loss.append(vloss_dict['perceptual'])
                    val_pbar.update(1)

                val_pbar.close()
                avg_val_loss = (
                    accelerator.gather(
                        torch.tensor(val_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_val_lpips = (
                    accelerator.gather(
                        torch.tensor(val_lpips_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_val_psnr = (
                    accelerator.gather(
                        torch.tensor(val_psnr, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                if accelerator.is_local_main_process:
                    for tag, val in [
                        ("val/loss", avg_val_loss),
                        ("val/lpips", avg_val_lpips),
                        ("val/psnr", avg_val_psnr),
                    ]:
                        writer.add_scalar(tag, val, global_step)
                vae.encoder.train()

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("train/loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)