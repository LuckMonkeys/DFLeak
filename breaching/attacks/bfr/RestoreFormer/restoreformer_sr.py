
import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pdb

# sys.path.append(os.getcwd())

from .main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange, tqdm

import cv2
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import torchvision
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, imwrite, tensor2img

def cv2tensor(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).float() 
    
    return tensor

def restoration(model,
                face_helper,
                img,
                # save_root,
                has_aligned=False,
                only_center_face=True,
                suffix=None,
                paste_back=False, 
                device="cuda:0"):

    img = torchvision.transforms.ToPILImage()(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
    ori_img_shape = img.shape
    face_helper.clean_all()


    if has_aligned:
        input_img = cv2.resize(img, (512, 512))
        face_helper.cropped_faces = [input_img]
    else:
        raise NotImplementedError("The image must be aligned")

    # face restoration
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to('cuda')

        try:
            with torch.no_grad():
                output = model(cropped_face_t)
                restored_face = tensor2img(output[0].squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = cropped_face

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)


    # breakpoint() #  NOTE: check the restored face value 

    if restored_face.shape[0] != ori_img_shape[0]:
        restored_face = cv2.resize(restored_face, (ori_img_shape[0], ori_img_shape[1]), interpolation=cv2.INTER_LINEAR)

    restored_face = cv2tensor(restored_face) / 255.0 # NOTE: check the restored face value, [0,1]
    return restored_face



def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        keys = list(sd.keys())

        state_dict = model.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        un_pretrained_keys = []
        for k in require_keys:
            if k not in keys: 
                # miss 'vqvae.'
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd[k]

        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')

        model.load_state_dict(state_dict, strict=True)

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def load_model_and_dset(config, ckpt, gpu, eval_mode):

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}

    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return model


def restoreformer_sr(img_lq, gpu_id=0, opt_resume="./breaching/attacks/bfr/RestoreFormer/experiments/RestoreFormer/last.ckpt", opt_base=[],  opt_config="./breaching/attacks/bfr/RestoreFormer/configs/RestoreFormer.yaml", opt_ignore_base_data=False, opt_top_k=100, opt_temperature=1.0, opt_upscale_factor=1, opt_test_path="./inputs/whole_imgs", opt_suffix=None, opt_only_center_face=False, opt_aligned=True, opt_paste_back=False, **kwargs): 

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt_resume:
        if not os.path.exists(opt_resume):
            raise ValueError("Cannot find {}".format(opt_resume))
        if os.path.isfile(opt_resume):
            paths = opt_resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt =opt_resume

        else:
            assert os.path.isdir(opt_resume), opt_resume
            logdir = opt_resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt_base = base_configs+opt_base

    if opt_config:
        if type(opt_config) == str:
            if not os.path.exists(opt_config):
                raise ValueError("Cannot find {}".format(opt_config))
            if os.path.isfile(opt_config):
                opt_base = [opt_config]
            else:
                opt_base = sorted(glob.glob(os.path.join(opt_config, "*-project.yaml")))
        else:
            opt_base = [opt_base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt_base]
    # cli = OmegaConf.from_dotlist(unknown)
    if opt_ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    # config = OmegaConf.merge(*configs, cli)
    config = OmegaConf.merge(*configs)
    
    print(config)
    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    model = load_model_and_dset(config, ckpt, gpu, eval_mode)
    
    face_helper = FaceRestoreHelper(
        opt_upscale_factor, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png')


    img_hq_list = [] 
    for img in img_lq:
        img_t = restoration(
                model,
                face_helper,
                img,
                # outdir,
                has_aligned=opt_aligned,
                only_center_face=opt_only_center_face,
                suffix=opt_suffix,
                paste_back=opt_paste_back,
                device=device)
        img_hq_list.append(img_t)
    
    img_hq = torch.stack(img_hq_list, dim=0)
    
    return img_hq

