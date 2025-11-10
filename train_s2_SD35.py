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
from core.utils.tabulate import tabulate
from core.model.other_impls import SD3Tokenizer, SDXLClipG, SDClipModel, T5XXLModel

import matplotlib.pyplot as plt


def print_vram_state(msg, logger=None):
    alloc = torch.cuda.memory_allocated() / 1024**3
    cache = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if logger:
        logger.info(
            f"[GPU memory]: {msg}, allocated = {alloc:.2f} GB, "
            f"cached = {cache:.2f} GB, peak = {peak:.2f} GB"
        )
    return alloc, cache, peak


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device=device,
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", device, torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float16):
        with safe_open(
            f"{model_folder}/t5xxl_fp16.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


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

def process_in(latent): return (latent - vae_shift_factor) * vae_scale_factor
def process_out(latent): return (latent / vae_scale_factor) + vae_shift_factor
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
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape

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

        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0],
            },
        }
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

    def apply_model(self, x, sigma, c_crossattn=None, y=None, skip_layers=[], controlnet_cond=None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        controlnet_hidden_states = None
        model_output = self.diffusion_model(
            x.to(dtype),
            timestep,
            context=c_crossattn.to(dtype),
            y=y.to(dtype),
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
def fix_cond(cond):
    cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
    return {"c_crossattn": cond, "y": pooled}


def build_text_conditioners(textenc_dir, device, dtype=torch.float32):
    tokenizer = SD3Tokenizer()
    clip_l = ClipL(textenc_dir, device=device)
    clip_g = ClipG(textenc_dir, device=device)
    t5xxl  = T5XXL(textenc_dir, device=device, dtype=dtype)
    return tokenizer, clip_l, clip_g, t5xxl


@torch.no_grad()
def encode_prompt(prompt, tokenizer, clip_l, clip_g, t5xxl, device):
    tokens = tokenizer.tokenize_with_weights(prompt)
    l_out, l_pooled = clip_l.model.encode_token_weights(tokens["l"])
    g_out, g_pooled = clip_g.model.encode_token_weights(tokens["g"])
    t5_out, t5_pooled = t5xxl.model.encode_token_weights(tokens["t5xxl"])
    lg_out = torch.cat([l_out, g_out], dim=-1)
    lg_out = F.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
    cond = torch.cat([lg_out, t5_out], dim=-2).to(device)
    pooled = torch.cat([l_pooled, g_pooled], dim=-1).to(device)
    return cond.half(), pooled.half()



def prepare_sd35_inputs(pred_latent, sd3_model: SD3, cond, pooled, device, weight_dtype, model_t=200):
    B = pred_latent.size(0)
    x = pred_latent.to(device, weight_dtype)
    t = torch.full((B,), model_t, device=device, dtype=torch.float32)
    sigma = sd3_model.model.model_sampling.sigma(t)
    return dict(x=x, sigma=sigma, c_crossattn=cond, y=pooled)



# Pretty tabulated model summary
def summary_models(modelDict):
    rows = []
    for name, m in modelDict.items():
        if not isinstance(m, torch.nn.Module): continue
        tot = sum(p.numel() for p in m.parameters()) / 1e6
        learn = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
        rows.append([name, type(m).__name__, f"{tot:.2f}", f"{learn:.2f}"])
    print(tabulate(rows, headers=["Model", "Type", "Params (M)", "Trainable (M)"], tablefmt="pretty"))


# -----------------------------
# Main
# -----------------------------
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
        ckpt_dir = os.path.join(exp_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[+] Experiment dir: {exp_dir}")

    # --- Text encoders for conditioning (empty prompt default) ---
    tokenizer, clip_l, clip_g, t5xxl = build_text_conditioners(cfg.train.textenc_dir, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        cond_blank, pooled_blank = encode_prompt("", tokenizer, clip_l, clip_g, t5xxl, device=device)
        print("[+] Prepared blank text conditioning")
        print(f"Cond shape: {cond_blank.shape}, pooled shape: {pooled_blank.shape}")
    
    # Remove tokenizer, clip_l, clip_g, t5xxl from VRAM memory
    del tokenizer, clip_l, clip_g, t5xxl
    torch.cuda.empty_cache()

    # Create & load VAE from pretrained SD 3.5 model:
    MODEL = f"{cfg.train.textenc_dir}/sd3.5_large.safetensors" 
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
    sd3 = SD3(MODEL, shift=3.0, device="cpu", verbose=False)
    # sd3.model.eval().requires_grad_(False)

    # Add LoRA params to latent transformer
    target_modules = cfg.train.lora_modules
    print(f"[+] Add lora parameters to {target_modules}")
    G_lora_cfg = LoraConfig(
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_rank, 
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    sd3.model = get_peft_model(sd3.model, G_lora_cfg)
    lora_params = list(filter(lambda p: p.requires_grad, sd3.model.parameters()))
    assert lora_params, "Failed to find lora parameters"
    print(f"\n[+] LoRA trainable params: {sum(p.numel() for p in lora_params) / 1000000} M\n")
    for p in filter(lambda p: p.requires_grad, sd3.model.parameters()):
        p.data = p.to(torch.float16)

    
    # Load Discriminator & make trainable
    D = ImageConvNextDiscriminator().to(device=accelerator.device)
    D.train().requires_grad_(True)



    # Setup optimizers:
    G_params = [p for p in sd3.model.parameters() if p.requires_grad]
    D_params = [p for p in D.parameters() if p.requires_grad]
    G_opt = torch.optim.AdamW(G_params, lr=cfg.train.learning_rate, betas=(0.9, 0.99))
    D_opt = torch.optim.AdamW(D_params, lr=cfg.train.learning_rate, betas=(0.9, 0.99))

    summary_models({"Generator(MM-DiTX+LoRA)": sd3.model, "Discriminator": D, "VAE": vae})
    
    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True, 
        pin_memory=True
    )
    batch_transform = instantiate_from_config(cfg.batch_transform)
    # val_dataset = instantiate_from_config(cfg.dataset.val)
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=cfg.train.batch_size,
    #     num_workers=cfg.train.num_workers,
    #     shuffle=False,
    #     drop_last=False,
    # )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")


    # Prepare models for training/inference:
    vae.to(device)
    sd3.model.to(device)
    sd3.model, D, G_opt, D_opt, loader = accelerator.prepare(
        sd3.model, D, G_opt, D_opt, loader
    )
    sd3.model = accelerator.unwrap_model(sd3.model)

    with warnings.catch_warnings():
        # avoid warnings from lpips internal
        warnings.simplefilter("ignore")
        lpips_model = (
            lpips.LPIPS(net="vgg", verbose=accelerator.is_local_main_process)
            .eval()
            .to(device)
        )

    if accelerator.is_local_main_process:
        writer = SummaryWriter(cfg.train.exp_dir)
        print(f"[+] Train steps: {cfg.train.train_steps:,}")
        print(f"[+] Dataset: {len(dataset):,} samples")

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    use_autocast = accelerator.mixed_precision != "no"
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    # Training loop:
    while global_step < max_steps:
        pbar = tqdm(total=len(loader), disable= not accelerator.is_local_main_process, unit="batch")
        do_G = True
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, _, _ = batch
            
            gt = rearrange(batch["gt"].float(), "b h w c -> b c h w").contiguous()
            lq = rearrange(batch["lq"].float(), "b h w c -> b c h w").contiguous()

            with torch.no_grad():
                z_lq = vae.encode(lq).mode()        # [B,16,H/8,W/8] for SD3.5 latent VAE
            z_in = process_in(z_lq)                  # scale to model input space
            
            # Prepare SD3.5 inputs (one-step, t=200)
            inp = prepare_sd35_inputs(
                pred_latent=z_in,
                sd3_model=sd3,
                cond=cond_blank, 
                pooled=pooled_blank,
                device=device, 
                weight_dtype=torch.float16, 
                model_t=200
            )

            if do_G:
                D.eval().requires_grad_(False)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                    denoised = sd3.model.apply_model(
                        x=inp["x"], sigma=inp["sigma"], c_crossattn=inp["c_crossattn"], y=inp["y"]
                    )
                    z_out = process_out(denoised.float())
                    xhat = vae.decode(z_out).float().clamp(-1, 1)

                    # Losses
                    l2 = F.mse_loss(xhat, gt) * cfg.train.loss_scales.lambda_l2
                    lp = lpips_model(xhat, gt).mean() * cfg.train.loss_scales.lambda_lpips
                    gan = D(xhat, for_G=True).mean() * cfg.train.loss_scales.lambda_gan
                    loss_G = l2 + lp + gan

                accelerator.backward(loss_G)
                torch.nn.utils.clip_grad_norm_(sd3.model.parameters(), 1.0)
                G_opt.step(); G_opt.zero_grad()
            
                # Log something
                loss_dict = {"G/total": loss_G.detach(), "G/l2": l2.detach(), "G/lpips": lp.detach(), "G/gan": gan.detach()}
                do_G = False
      
            else:  
                # --- Discriminator step ---
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                        denoised = sd3.model.apply_model(
                            x=inp["x"], sigma=inp["sigma"], c_crossattn=inp["c_crossattn"], y=inp["y"]
                        )
                        z_out = process_out(denoised.float())
                        xhat = vae.decode(z_out).float().clamp(-1, 1)

                D.train().requires_grad_(True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                    loss_D_real, real_logits = D(gt, for_real=True, return_logits=True)
                    loss_D_fake, fake_logits = D(xhat, for_real=False, return_logits=True)
                    loss_D = loss_D_real.mean() + loss_D_fake.mean()

                accelerator.backward(loss_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                D_opt.step(); D_opt.zero_grad()

                # Optional: summarize logits (avoid big tensors in TB)
                with torch.no_grad():
                    rl = torch.stack([m.mean() for m in real_logits]).mean() if isinstance(real_logits, (list, tuple)) else torch.as_tensor(0.0, device=device)
                    fl = torch.stack([m.mean() for m in fake_logits]).mean() if isinstance(fake_logits, (list, tuple)) else torch.as_tensor(0.0, device=device)
                loss_dict = {"D/total": loss_D.detach(), "D/real_logit": rl, "D/fake_logit": fl}
                do_G = True

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"epoch {epoch} | step {global_step}")

            _, _, peak = print_vram_state(None)
            pbar.set_description(f"Generator: {do_G}, VRAM peak: {peak:.2f} GB")

            # Log losses:
            if accelerator.is_local_main_process and (global_step % cfg.train.log_every == 0 or global_step == 1):
                for k, v in loss_dict.items():
                    writer.add_scalar(k, v.item(), global_step)

            # Images
            if accelerator.is_local_main_process and (global_step % cfg.train.image_every == 0 or global_step == 1):
                N = min(4, gt.size(0))
                to_log = [
                    ("image/pred", (xhat[:N] + 1) / 2),
                    ("image/gt",   (gt[:N]  + 1) / 2),
                    ("image/lq",   (lq[:N]  + 1) / 2),
                ]
                for tag, img in to_log:
                    writer.add_image(tag, make_grid(img, nrow=N), global_step)


            # Save checkpoint:
            if accelerator.is_main_process and (global_step % cfg.train.ckpt_every == 0):
                save_path = os.path.join(cfg.train.exp_dir, "checkpoints", f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                print(f"[+] Saved: {save_path}")

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