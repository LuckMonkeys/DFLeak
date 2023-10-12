from omegaconf import OmegaConf
# import guided_diffusion
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import traceback

args = OmegaConf.load("./breaching/attacks/accelarate/guided_diffusion/gdiffusion.yaml").diffusion


##get the dir name from imagenet25
with open("../data/imagenet25/imagefolder_classes.txt", 'r') as f:
    content = f.readlines()[0]
    dir_name_list = content.strip().split(',')
##get imagenet classes and index 
import json
f = open("../data/imagenet25/imagenet_class_index.json")
content = json.load(f)
dir2index = {}
for key, value in content.items():
    dir2index[value[0]] = key


def reload_model(model, ckpt):
    from collections import OrderedDict
    if list(model.state_dict().keys())[0].startswith('module.'):
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = ckpt
        else:
            ckpt = OrderedDict({f'module.{key}':value for key, value in ckpt.items()})
    else:
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        else:
            ckpt = ckpt
    model.load_state_dict(ckpt)
    
    
def normalize_th(im, mean=0.5, std=0.5, reverse=False):
    '''
    Input:
        im: b x c x h x w, torch tensor
        Normalize: (im - mean) / std
        Reverse: im * std + mean

    '''
    if not isinstance(mean, (list, tuple)):
        mean = [mean, ] * im.shape[1]
    mean = torch.tensor(mean, device=im.device).view([1, im.shape[1], 1, 1])

    if not isinstance(std, (list, tuple)):
        std = [std, ] * im.shape[1]
    std = torch.tensor(std, device=im.device).view([1, im.shape[1], 1, 1])

    if not reverse:
        out = (im - mean) / std
    else:
        out = im * std + mean
    return out

# NUM_CLASSES, model_and_diffusion_defaults, classifier_defaults, create_model_and_diffusion, create_classifier, add_dict_to_argparser, args_to_dict = guided_diffusion.NUM_CLASSES, guided_diffusion.model_and_diffusion_defaults, guided_diffusion.classifier_defaults, guided_diffusion.create_model_and_diffusion, guided_diffusion.create_classifier, guided_diffusion.add_dict_to_argparser, guided_diffusion.args_to_dict

from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def gdiffusion_sr(img_lq, gpu_id=0, start_timesteps=100, timestep_respacing="250", post_fn=None, **kwargs):
    # print(args)
    assert start_timesteps < int(timestep_respacing) 
    if post_fn is None:
        post_fn = lambda x: normalize_th(
                im=x,
                mean=0.5,
                std=0.5,
                reverse=False,
                )
    
    diff_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    reload_model(diff_model, torch.load(args.model_path))
    setup = {"device":torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")}

    
    diff_model.to(device=setup['device'])
    if args.use_fp16:
        diff_model.convert_to_fp16()
    diff_model.eval()
    
    
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        torch.load(args.classifier_path)
    )
    classifier.to(device=setup['device'])
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    
    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return diff_model(x, t, y if args.class_cond else None) 
    
    # sample_fn = (
    #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    # )
    
    # labels = torch.randint( low=0, high=25, size=(img_lq.shape[0],), device=setup['device'])
    # breakpoint()
    # labels = torch.tensor([0,0,0], device=setup['device'])
    ##change label from imagenet25 to real imagenet label
    labels = kwargs.get("aux_info") 
    # breakpoint()
    assert labels is not None, "Please input labels for guided diffusion model"

    imagenet_labels = []
    for label in labels:
        dirname = dir_name_list[label.item() if isinstance(label, torch.Tensor) else label]
        try:
            imagenet_labels.append(int(dir2index[dirname]))
        except:
            traceback.print_exc()
            # print(e)
            print(dir2index)

    if len(imagenet_labels) != img_lq.shape[0]:
        imagenet_labels = imagenet_labels * img_lq.shape[0]
    # breakpoint()
    labels = torch.tensor(imagenet_labels, device=setup['device'])
    # print("original ImageNet labels:", labels)
    
    model_kwargs = {}
    model_kwargs['y'] = labels
    
    #resize img_hq
    h_lq, w_lq = img_lq.shape[2:4]
    if not (h_lq == args.image_size and w_lq == args.image_size):
        # im_hq = resize(im_hq, out_shape=(self.configs.im_size,) * 2).to(torch.float32)
        img_lq_resize = torchvision.transforms.Resize((args.image_size, args.image_size))(img_lq)
    else:
        img_lq_resize = img_lq
    
    # diffuse for im_lq
    if start_timesteps is None:
        start_timesteps = diffusion.num_timesteps

    yt = diffusion.q_sample(
            x_start=post_fn(img_lq_resize),
            t=torch.tensor([start_timesteps,]*img_lq_resize.shape[0], device=setup['device']),
            )
    
    
    if 'ddim' in args.timestep_respacing:
        sample = diffusion.ddim_sample_loop(
                # model_fn,
                diff_model,
                shape=yt.shape,
                noise=yt,
                start_timesteps=start_timesteps,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=None,
                progress=False,
                eta=0.0,
                )
    else:
        sample = diffusion.p_sample_loop(
                # model_fn,
                diff_model,
                shape=yt.shape,
                noise=yt,
                start_timesteps=start_timesteps,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=None,
                progress=False,
                )
     
    
    sample = normalize_th(sample,mean=0.5, std=0.5, reverse=True).clamp(0.0, 1.0)

    if not (h_lq == args.image_size and w_lq == args.image_size):
        sample = torchvision.transforms.Resize((h_lq, w_lq))(sample)

    return sample # [0,1] 'rgb
    

# if "__main__" == __name__:
#     gdiffusion_sr()
   