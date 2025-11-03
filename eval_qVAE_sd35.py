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
import numpy as np
from PIL import Image
# Debugging libs: ###############
# import matplotlib.pyplot as plt
# import numpy as np
#################################

from core.utils.common import instantiate_from_config, calculate_psnr_pt, to

import matplotlib.pyplot as plt


import piq
def compute_full_reference_metrics(gt_img, out_img):
    # PSNR SSIM LPIPS
    
    # print(gt_img.shape, out_img.shape)
    # print(gt_img.max(), out_img.max(), gt_img.min(), out_img.min())
    # print("Full-reference scores:")
    psnr = piq.psnr(out_img, gt_img, data_range=1., reduction='none')
    # print(f"PSNR: {psnr.item():.2f} dB")

    ssim = piq.ssim(out_img, gt_img, data_range=1.) 
    # print(f"SSIM: {ssim.item():.4f}")

    lpips = piq.LPIPS(reduction='none')(out_img, gt_img)
    # print(f"LPIPS: {lpips.item():.4f}")
    return psnr.item(), ssim.item(), lpips.item()

import pyiqa
# from DeQAScore.src import Scorer
def compute_no_reference_metrics(out_img):
    # center crop to 224x224 for no-reference metrics
    _, _, h, w = out_img.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    out_img = out_img[:, :, top:top+224, left:left+224]

    # ManIQA DeQA MUSIQ ClipIQA
    maniqa = pyiqa.create_metric('maniqa', device=torch.device("cuda:1"))
    clipiqa = pyiqa.create_metric('clipiqa', device=torch.device("cuda:1"))
    musiq = pyiqa.create_metric('musiq', device=torch.device("cuda:1"))
    # deqa = Scorer(model_type='deqa', device=torch.device("cuda:1"))

    maniqa_score = maniqa(out_img).item()
    clipiqa_score = clipiqa(out_img).item()
    musiq_score = musiq(out_img).item()
    # deqa_score = deqa.score([Image.fromarray((out_img.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))])[0]

    # print("No-reference scores:")
    # print(f"ManIQA: {maniqa_score:.4f}")
    # print(f"ClipIQA: {clipiqa_score:.4f}")
    # print(f"MUSIQ: {musiq_score:.4f}")
    # print(f"DeQA: {deqa_score:.4f}")
    return maniqa_score, clipiqa_score, musiq_score #, deqa_score


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

def tensor2image(img_tensor):
        return (
            (img_tensor * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .squeeze()
            .numpy()
        )


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(310)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
   
    if accelerator.is_main_process:
        exp_dir = cfg.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Eval directory created at {exp_dir}")


    # Create & load VAE from pretrained SD 3.5 model:
    MODEL = cfg.sd_path
    weights = torch.load(MODEL, map_location="cpu")
    vae = SDVAE(device="cpu", dtype=torch.float32)
    vae.load_state_dict(weights, strict=True)
    vae.requires_grad_(False)
    vae.to(device)
    print(f"[+] Loaded SD35 qVAE successfully and moved to {device}")
    # Setup data:
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.dataset.val.batch_size,
        num_workers=cfg.dataset.val.num_workers,
        shuffle=False,
        drop_last=False,
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(val_dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)


    # Val loop:
    val_ssim = []
    val_lpips_loss = []
    val_psnr = []
    man = []
    clip = []
    musiq_list = []
    iter_idx = 0
    for val_batch in tqdm(val_loader):
        to(val_batch, device)
        val_batch = batch_transform(val_batch)
        val_gt, val_lq, val_prompt, val_gt_path = val_batch
        val_gt = ((
            rearrange(val_gt, "b h w c -> b c h w")
            .contiguous()
            .float()
        ) + 1.) / 2.
        val_lq = (
            rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
        )
        with torch.no_grad():
            out = vae.decode(vae.encode(val_lq.to(device))).clamp(0,1).float()
        psnr, ssim, lpips = compute_full_reference_metrics(val_gt.to(device), out)
        maniqa, clipiqa, musiq = compute_no_reference_metrics(out)

        val_ssim.append(ssim)
        val_lpips_loss.append(lpips)
        val_psnr.append(psnr)
        man.append(maniqa)
        clip.append(clipiqa)
        musiq_list.append(musiq)

        # print(out.size(), val_gt.size(), val_lq.size())
        out_img = tensor2image(out.detach().cpu())
        gt_img = tensor2image(val_gt.cpu())
        inp_img = tensor2image( ( (val_lq + 1.) / 2. ).detach().cpu() ) 

        Image.fromarray(gt_img).save(os.path.join(cfg.exp_dir, f"gt_{'color' if cfg.color else 'mono'}_{str(iter_idx).zfill(4)}.png"))
        Image.fromarray(inp_img).save(os.path.join(cfg.exp_dir, f"lq_{'color' if cfg.color else 'mono'}_{str(iter_idx).zfill(4)}.png"))
        Image.fromarray(out_img).save(os.path.join(cfg.exp_dir, f"out_{'color' if cfg.color else 'mono'}_{str(iter_idx).zfill(4)}.png"))

        iter_idx += 1

      
    with open(f"/nobackup1/aryan/results/evaluation_SD35_{'color' if cfg.color else 'mono'}_Stage1.txt", "w") as f:
        f.write("Overall scores on full_test_set:\n")
        f.write("---------- FR scores ----------\n")
        f.write(f"Average PSNR: {np.mean(val_psnr):.3f} dB\n")
        f.write(f"Average SSIM: {np.mean(val_ssim):.4f}\n")
        f.write(f"Average LPIPS: {np.mean(val_lpips_loss):.4f}\n")
        f.write("---------- NR scores ----------\n")
        f.write(f"Average ManIQA: {np.mean(man):.4f}\n")
        f.write(f"Average ClipIQA: {np.mean(clip):.4f}\n")
        f.write(f"Average MUSIQ: {np.mean(musiq_list):.4f}\n")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)