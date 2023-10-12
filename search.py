import torch
from breaching.cases.data.datasets_vision import _build_dataset_vision, _parse_data_augmentations
from omegaconf import OmegaConf
import random

img_scale = 112
base_dir = ".."
num_sample= 100
target_cfg_data = {'db': {'name': None}, 'name': 'bFFHQ_Gender', 'modality': 'vision', 'task': 'classification', 'path': '../data', 'size': 19200, 'classes': 2, 'scale': 112, 'shape': [3, img_scale, img_scale], 'normalize': True, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201], 'augmentations_train': {'RandomResizedCrop': img_scale, 'RandomHorizontalFlip': 0.5}, 'augmentations_val': {'Resize': img_scale}, 'augmentations_ats': {'policy':None}, 'default_clients': 100, 'partition': 'random', 'examples_from_split': 'valid', 'batch_size': 128, 'caching': False}


celebahq_cfg_data = {'db': {'name': None}, 'name': 'CelebaHQ_Gender', 'modality': 'vision', 'task': 'classification', 'path': '../data', 'size': 30000, 'classes': 2, 'scale': 112, 'shape': [3, img_scale, img_scale], 'normalize': True, 'mean': [0.506, 0.425, 0.382], 'std': [0.265, 0.245, 0.241], 'augmentations_train': {'RandomResizedCrop': img_scale, 'RandomHorizontalFlip': 0.5}, 'augmentations_val': {'Resize': img_scale}, 'augmentations_ats': {'policy':None}, 'default_clients': 100, 'partition': 'random', 'examples_from_split': 'valid', 'batch_size': 128, 'caching': False}

lfw_cfg_data = {'db': {'name': None}, 'name': 'LFWA_Gender', 'modality': 'vision', 'task': 'classification', 'path': '/home/zx/data', 'size': 13000, 'classes': 2, 'scale': 112, 'shape': [3, img_scale, img_scale], 'normalize': True, 'mean': [0.439, 0.383, 0.342], 'std': [0.297, 0.273, 0.268], 'augmentations_train': {'RandomResizedCrop': img_scale, 'RandomHorizontalFlip': 0.5}, 'augmentations_val': {'Resize': img_scale}, 'default_clients': 100, 'partition': 'random', 'examples_from_split': 'valid', 'batch_size': 128, 'caching': False}



ori_cfg_data = celebahq_cfg_data #  or lfw_cfg_data

target_cfg_data = OmegaConf.create(target_cfg_data)
ori_cfg_data = OmegaConf.create(ori_cfg_data)

#load model and loss_fn
from breaching.cases.models.model_preparation import construct_model
cfg_model = "ResNet18"
state_dict_path = "" #provide the trained model


state_dict = torch.load(state_dict_path)
model, loss_fn = construct_model(cfg_model=cfg_model, cfg_data=ori_cfg_data, pretrained=False)
model.model.load_state_dict(state_dict)



#load target dataset
target_dataset, collect_fn = _build_dataset_vision(cfg_data=target_cfg_data, split='train')
target_dataset.transform = None # do not set the transformation for target images

####load sample from dataset
sample_list = random.sample(range(len(target_dataset)), num_sample)  
ori_transform = _parse_data_augmentations(cfg_data=target_cfg_data, split="valid") # only use valid transform for random images


def compute_feature(gradients, labels):
    import torch 
    weights = gradients[-2]
    bias = gradients[-1]
    grads_fc_debiased = weights / bias[:, None]
    features_per_label = []
    for label in labels:
        if bias[label] != 0:
            features_per_label.append(grads_fc_debiased[label])
        else:
            features_per_label.append(torch.zeros_like(grads_fc_debiased[0]))
    return torch.stack(features_per_label)



class _LinearFeatureHook:
    """Hook to retrieve input to given module."""

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        input_features = input[0]
        self.features = input_features

    def close(self):
        self.hook.remove()


for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        target_refs = _LinearFeatureHook(module)


def get_closest_img(num_sample, gradient_dir, target_dataset, sample_list, ori_transform, reverse=False):
    closest_imgs_idx = []
    for idx, in range(num_sample):
       
        #read gradients from dir 
        ori_gradients = torch.load(os.path.join(gradient_dir, f"gradients_{idx}.pth"))
        ori_features = compute_feature(ori_gradients, [0])

        distance = []

        for target_img_idx in sample_list:
            target_img, target_label = target_dataset[target_img_idx]

            transform_target_img = ori_transform(target_img)
            
            target_loss =loss_fn(model(transform_target_img.unsqueeze(0)), torch.tensor([target_label]))
            target_features = target_refs.features

            ##sort the distance
            distance.append((ori_features - target_features.to(ori_features.device)).pow(2).mean()) 
       
        closest_imgs_idx.append(sample_list[torch.argmin(torch.tensor(distance)).item()]) 
    return closest_imgs_idx
            
#save the image

import os
gradient_dir =f"{base_dir}/breaching/gradients/celebahq"
# gradint_dir =f"{base_dir}/breaching/gradients/lfwa"


closest_imgs_idx = get_closest_img(num_sample=num_sample, gradient_dir=gradient_dir, target_dataset=target_dataset, sample_list=sample_list, ori_transform=ori_transform)
save_dir = f"{base_dir}/breaching/out/celeba_hq/search/112"
# save_dir = f"{base_dir}/breaching/out/lfw/search/112"

from PIL import Image
import os
for idx, img_idx in enumerate(closest_imgs_idx):
    target_img, _ = target_dataset[img_idx]
    if isinstance(target_img, torch.Tensor):
        import torchvision
        
        target_img = torchvision.transforms.ToPILImage()(target_img)
    if not isinstance(target_img, Image.Image):
        raise TypeError("Target image should be PIL Image object")
    user_save_dir = os.path.join(os.path.join(save_dir, f"user{idx}"))
    if not os.path.exists(user_save_dir):
        os.makedirs(user_save_dir, exist_ok=True)
    target_img.save(os.path.join(user_save_dir, "0.png"))
