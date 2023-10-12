"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os,sys
sys.path.insert(0, './')

import numpy as np
import torch as th
# import torch.distributed as dist
import torch.nn.functional as F

# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

device = th.device('cuda:0')
def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure('/home/zx/data/GitRepo/guided-diffusion/log')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        # dist_util.load_state_dict(args.model_path, map_location="cpu")
        th.load(args.model_path)
    )
    model.to(device=device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        # dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        th.load(args.classifier_path)
    )
    classifier.to(device)
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    
    ## load existing image from attack and resize to 256, do normalized
    import torchvision
    from natsort import os_sorted
    from PIL import Image


    # img = th.randn_like(img)

    while len(all_images) * args.batch_size< args.num_samples:
        model_kwargs = {}
        classes = th.tensor([24]*args.batch_size, device=device)
        # classes = th.randint(
        #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
        # )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            # cond_fn=None,
            # device=dist_util.dev(),
            device=device, 
            # noise=img
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.append(sample.cpu().numpy())
        all_labels.append(classes.cpu().numpy())

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_labels, classes)
        # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
