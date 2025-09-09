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
from peft import LoraConfig, get_peft_model

import math
import re

# Debugging libs: ###############
# import matplotlib.pyplot as plt
# import numpy as np
#################################

from core.utils.common import instantiate_from_config, calculate_psnr_pt, to
from core.model.mmditx import MMDiTX
from core.model.convnext import ImageConvNextDiscriminator
from core.model.other_impls import SD3Tokenizer

import matplotlib.pyplot as plt

def compute_loss(gt, z_gt, z_pred, xhat_gt, xhat_lq, lpips_model, loss_mode, scales):
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
        elif "gt" in loss_mode:
            gt_loss = scales.gt * F.l1_loss(xhat_gt, gt, reduction="mean")
            loss_dict["gt_loss"] = gt_loss.item()
        if "perceptual" in loss_mode:
            perceptual_loss = (scales.perceptual_gt * lpips_model(xhat_gt, gt)) + \
                (scales.perceptual_lq * lpips_model(xhat_lq, gt))
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
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    }
}
#################################################################

#################################################################
# Latent Space DiT
#################################################################
class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image
    

class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""

    def __init__(
        self,
        shift=1.0,
        device=None,
        dtype=torch.float32,
        file=None,
        prefix="",
        control_model_ckpt=None,
        verbose=False,
    ):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        # context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape

        qk_norm = (
            "rms"
            if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in file.keys()
            else None
        )
        x_block_self_attn_layers = sorted(
            [
                int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
                for key in list(
                    filter(
                        re.compile(".*.x_block.attn2.ln_k.weight").match, file.keys()
                    )
                )
            ]
        )

        # context_embedder_config = {
        #     "target": "torch.nn.Linear",
        #     "params": {
        #         "in_features": context_shape[1],
        #         "out_features": context_shape[0],
        #     },
        # }
        context_embedder_config = None  # No text encoder
        self.diffusion_model = MMDiTX(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            in_channels=16,
            depth=depth,
            num_patches=num_patches,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            qk_norm=qk_norm,
            x_block_self_attn_layers=x_block_self_attn_layers,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)
        self.control_model = None

    def apply_model(self, x, sigma, c_crossattn=None, y=None, skip_layers=[]):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        controlnet_hidden_states = None
    
        model_output = self.diffusion_model(
            x.to(dtype),
            timestep,
            context=c_crossattn,
            y=y,
            controlnet_hidden_states=controlnet_hidden_states,
            skip_layers=skip_layers,
        ).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype
    

class SD3:
    def __init__(self, model, shift=3.0, verbose=False, device="cpu"):

        with safe_open(model, framework="pt", device="cpu") as f:
            control_model_ckpt = None
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cpu",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cpu", torch.float16)
    
#################################################################

def prepare_sd35_inputs(pred_latent, model_sampling, device, weight_dtype, model_t=200):
    """
    Returns the exact kwargs your BaseModel.apply_model expects:
      x:           latent in model input space (after process_in)
      sigma:       [B] noise levels (mapped from model_t via model_sampling.sigma)
      c_crossattn: context embeddings (text) [B, T, C]
      y:           pooled text embedding [B, C]
    """
    B = pred_latent.size(0)

    # 2) Encode image to latent, then into model’s input space
    x = pred_latent.to(device, weight_dtype)                # SD3.5 LT is fp32 

    # 3) Sigma schedule (discrete flow mapping)
    # model_sampling.sigma expects a tensor timestep (1..1000) scaled inside. Set to 200 as in HYPIR
    t = torch.full((B,), model_t, device=device, dtype=torch.float32)
    sigma = model_sampling.sigma(t)                  # [B], float32 is fine

    # 4) 
    c_crossattn = torch.zeros(B, 1, 2432, device=device, dtype=weight_dtype)  # [B, 0, D]
    y = torch.zeros(B, 2048, device=device, dtype=weight_dtype)               # [B, D]         

    return dict(x=x, sigma=sigma, c_crossattn=c_crossattn, y=y)


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(310)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    
    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")


    # Create & load VAE from pretrained SD 3.5 model:
    MODEL = "/home/argar/apgi/gqvr_sd35/pretrained_ckpt/sd3.5_large.safetensors" 
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        vae = SDVAE(device="cpu", dtype=torch.float32)
        prefix = ""
        if any(k.startswith("first_stage_model.") for k in f.keys()):
            prefix = "first_stage_model."
        load_into(f, vae, prefix, "cpu", torch.float32)

    # Load qVAE from checkpoint:
    vae_sd = torch.load(cfg.train.vae_path, map_location="cpu")
    vae.load_state_dict(vae_sd, strict=True)
    # Freeze VAE
    vae.eval().requires_grad_(False)

    # For future +prompt trainings
    # text_adapter = SD3Tokenizer()
    
    # LAtent Space DiT
    latent_transformer = SD3(MODEL)
    latent_transformer.model.eval().requires_grad_(False)
    # Add LoRA params to latent transformer
    target_modules = cfg.train.lora_modules
    print(f"[+] Add lora parameters to {target_modules}")
    G_lora_cfg = LoraConfig(
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_rank, # 768
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    latent_transformer.model = get_peft_model(latent_transformer.model, G_lora_cfg)
    lora_params = list(filter(lambda p: p.requires_grad, latent_transformer.model.parameters()))
    assert lora_params, "Failed to find lora parameters"
    for p in lora_params:
        p.data = p.to(torch.float16)

    print(f"\n[+] LoRA trainable params: {sum(p.numel() for p in lora_params) / 1000000} M\n")

    # Load Discriminator & make trainable
    D = ImageConvNextDiscriminator().to(device=accelerator.device)
    D.train().requires_grad_(True)

    # Setup optimizers:
    G_params = list(filter(lambda p: p.requires_grad, latent_transformer.model.parameters()))
    G_opt = torch.optim.AdamW(
        G_params,
        lr=cfg.train.learning_rate,
        betas=(0.9, 0.99)
    )

    D_params = list(filter(lambda p: p.requires_grad, D.parameters()))
    D_opt = torch.optim.AdamW(
        D_params,
        lr=cfg.train.learning_rate,
        betas=(0.9, 0.99)
    )

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
    latent_transformer.model.to(device)
    latent_transformer.model, D, G_opt, D_opt, loader, val_loader = accelerator.prepare(
        latent_transformer.model, D, G_opt, D_opt, loader, val_loader
    )
    latent_transformer.model = accelerator.unwrap_model(latent_transformer.model)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0


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
        generator_step = True # Alternate between G and D steps
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, _, _ = batch
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # Train step:
            # gt_latent = vae.encode(gt)
            pred_latent = vae.encode(lq)
            pred_latent = process_in(pred_latent)
            
            inp = prepare_sd35_inputs(
                    pred_latent=pred_latent,                
                    model_sampling=latent_transformer.model.model_sampling,
                    device=accelerator.device,
                    weight_dtype=torch.float16,
                    model_t=200  # Same as HYPIR
            )
            if generator_step:
                with torch.autocast(device_type=accelerator.device.type, dtype=torch.float16):
                    denoised = latent_transformer.model.apply_model(
                        x=inp["x"],
                        sigma=inp["sigma"],
                        c_crossattn=inp["c_crossattn"],
                        y=inp["y"],
                    )
                enhanced_latent = process_out(denoised.float())
                xhat_lq = vae.decode(enhanced_latent).clamp(-1, 1).float()

                accelerator.unwrap_model(D).eval().requires_grad_(False)
                loss_l2 = F.mse_loss(xhat_lq, gt, reduction="mean") * cfg.train.loss_scales.lambda_l2
                loss_lpips = lpips_model(xhat_lq, gt).mean() * cfg.train.loss_scales.lambda_lpips
                loss_disc = D(xhat_lq, for_G=True).mean() * cfg.train.loss_scales.lambda_gan
                loss_G = loss_l2 + loss_lpips + loss_disc
                accelerator.backward(loss_G)
                torch.nn.utils.clip_grad_norm_(latent_transformer.model.parameters(), max_norm=1.0)
                G_opt.step()
                G_opt.zero_grad()
                accelerator.wait_for_everyone()
            
                # Log something
                loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips, G_disc=loss_disc)
                generator_step = False
            else:  
                with torch.no_grad():
                    with torch.autocast(device_type=accelerator.device.type, dtype=torch.float16):
                        denoised = latent_transformer.model.apply_model(
                            x=inp["x"],
                            sigma=inp["sigma"],
                            c_crossattn=inp["c_crossattn"],
                            y=inp["y"],
                        )
                    enhanced_latent = process_out(denoised.float())
                    xhat_lq = vae.decode(enhanced_latent).clamp(-1, 1).float()

                accelerator.unwrap_model(D).train().requires_grad_(True)
                loss_D_real, real_logits = D(gt, for_real=True, return_logits=True)
                loss_D_fake, fake_logits = D(xhat_lq, for_real=False, return_logits=True)
                loss_D = loss_D_real.mean() + loss_D_fake.mean()
                accelerator.backward(loss_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                D_opt.step()
                D_opt.zero_grad()
                accelerator.wait_for_everyone()

                loss_dict = dict(D=loss_D)
                # logits = D(x) w/o sigmoid = log(p_real(x) / p_fake(x))

                with torch.no_grad():
                    real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=accelerator.device).mean()
                    fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=accelerator.device).mean()

                loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))

                generator_step = True    

            global_step += 1
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}"
            )


            # Log losses:
            if global_step % cfg.train.log_every == 0 or global_step % cfg.train.log_every == 0 == 1:
                if accelerator.is_local_main_process:
                    for k, v in loss_dict.items():
                        writer.add_scalar(f"train/{k}", v.item(), global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(cfg.train.exp_dir, "checkpoints", f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    print(f"Saved state to {save_path}")

            # Log images
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 2
                log_gt, log_lq = gt[:N], lq[:N]
                log_pred = xhat_lq[:N]
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/pred", (log_pred + 1) / 2),
                        ("image/gt", (log_gt + 1) / 2),
                        ("image/lq", (log_lq + 1) / 2),
                    ]:
                        writer.add_image(tag, make_grid(image, nrow=4), global_step)

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)