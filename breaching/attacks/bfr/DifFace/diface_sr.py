#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-02 20:43:41

import os
import torch
import argparse
import numpy as np
from pathlib import Path
# from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte

from .utils import util_opts
from .utils import util_image
from .utils import util_common
import torchvision

from .sampler import DifIRSampler
from .ResizeRight.resize_right import resize
from breaching.attacks.bfr.DifFace.basicsr.utils.download_util import load_file_from_url


align_img_size = 512



def diface_sr(img_lq, gpu_id=0, started_timesteps=100, timestep_respacing="250", aligned=True, draw_box=False, needsr=False, **kwargs):
        
#    img_lq the: tensor [0,1]

    cfg_path = './breaching/attacks/bfr/DifFace/configs/sample/iddpm_ffhq512_swinir.yaml'
    # cfg_path = '/home/zx/nfs/server3/breaching/breaching/attacks/bfr/DifFace/configs/sample/iddpm_ffhq512_swinir.yaml'
    # cfg_path = '/home/zx/breaching/breaching/attacks/bfr/DifFace/configs/sample/iddpm_ffhq512_swinir.yaml'

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = gpu_id
    configs.aligned = aligned
    assert started_timesteps < int(timestep_respacing)
    configs.diffusion.params.timestep_respacing = timestep_respacing

    # prepare the checkpoint
    if not Path(configs.model.ckpt_path).exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth",
            model_dir=str(Path(configs.model.ckpt_path).parent),
            progress=True,
            file_name=Path(configs.model.ckpt_path).name,
            )
    if not Path(configs.model_ir.ckpt_path).exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
            model_dir=str(Path(configs.model_ir.ckpt_path).parent),
            progress=True,
            file_name=Path(configs.model_ir.ckpt_path).name,
            )

    # build the sampler for diffusion
    sampler_dist = DifIRSampler(configs)

    import cv2 
    lq_img_size = img_lq.shape[2:] if len(img_lq.shape) == 4 else img_lq.shape[1:]
    img_lq = torchvision.transforms.Resize((align_img_size, align_img_size))(img_lq) #[0,1], 'rgb'
    if aligned:
        image_restored = sampler_dist.sample_func_ir_aligned(
                y0=img_lq,
                start_timesteps=started_timesteps,
                need_restoration=needsr,
                )[0] #[0,1], 'rgb'
    else:
        image_restored, face_restored, face_cropped = sampler_dist.sample_func_bfr_unaligned(
                y0=img_lq,
                start_timesteps=started_timesteps,
                need_restoration=needsr,
                draw_box=draw_box,
                ) #[0, 255], numpy
        
        image_restored = torch.from_numpy(image_restored).permute(2,0,1).float() / 255.0
    
    return torchvision.transforms.Resize(lq_img_size)(image_restored) #[0,1], 'rgb'

