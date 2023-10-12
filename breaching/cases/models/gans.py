import argparse
import os
import numpy as np
import math
import sys


import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from torch.autograd import Variable

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from ipywidgets import IntProgress



class Generator_32(nn.Module):
    def __init__(self, DIM=128):
        super(Generator_32, self).__init__()
        self.DIM = DIM
        
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, z):
        DIM = self.DIM
        output = self.preprocess(z)
        # print(output.shape)
        output = output.view(-1, 4 * DIM, 4, 4)
        # print(output.shape)
        output = self.block1(output)
        # print(output.shape)
        output = self.block2(output)
        # print(output.shape)
        output = self.deconv_out(output)
        # print(output.shape)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator_32(nn.Module):
    def __init__(self, DIM=128):
        super(Discriminator_32, self).__init__()
        
        self.DIM = DIM
        
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, img):
        DIM = self.DIM
        
        output = self.main(img)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

class Generator_56(nn.Module):
    def __init__(self, DIM=128):
        super(Generator_56, self).__init__()
        self.DIM = DIM
        
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 8 * DIM),
            nn.BatchNorm1d(4 * 4 * 8 * DIM),
            nn.ReLU(True),
            #state size. (ngf*16) x 4 x 4
        )
        self.main = nn.Sequential(
            # nn.Linear(128, 4 * 4 * 16 * DIM),
            # nn.BatchNorm1d(4 * 4 * 16 * DIM),
            # nn.ReLU(True),
            # #state size. (ngf*16) x 4 x 4
            
            nn.ConvTranspose2d(8 * DIM, 4 * DIM, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(4 * DIM),
            nn.ReLU(True),
            #state size. (ngf*4) x 7 x 7

            nn.ConvTranspose2d(4 * DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            #state size. (ngf*2) x 14 x 14

            nn.ConvTranspose2d(2 * DIM,  DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            #state size. (ngf*1) x 28 x 28
            
            # nn.ConvTranspose2d(2 * DIM, DIM, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(DIM),
            # nn.ReLU(True),
            # #state size. (ngf) x 56 x 56
            
            nn.ConvTranspose2d(DIM, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            #state size. (3) x 56 x 56
        )

        self.preprocess = preprocess

    def forward(self, z):
        DIM = self.DIM
        output = self.preprocess(z)
        # print(output.shape)
        output = output.view(-1, 8 * DIM, 4, 4)
        # print(output.shape)
        output = self.main(output)
        # print(output.shape)
        return output.view(-1, 3, 56, 56)


class Discriminator_56(nn.Module):
    def __init__(self, DIM=128):
        super(Discriminator_56, self).__init__()
        
        self.DIM = DIM
        
        main = nn.Sequential(
            #56x56
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #28x28
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #14x14
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #7x7
            nn.Conv2d(4 * DIM, 8 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #4x4
            # nn.Conv2d(8 * DIM, 16 * DIM, 3, 2, padding=1),
            # nn.LeakyReLU(),
            # #4x4
        )

        self.main = main
        self.linear = nn.Linear(8*4*4*DIM, 1)

    def forward(self, img):
        DIM = self.DIM
        
        output = self.main(img)
        # print(output.shape)
        output = output.view(-1, 8*4*4*DIM)
        output = self.linear(output)
        return output





class Generator_112(nn.Module):
    def __init__(self, DIM=128):
        super(Generator_112, self).__init__()
        self.DIM = DIM
        
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 16 * DIM),
            nn.BatchNorm1d(4 * 4 * 16 * DIM),
            nn.ReLU(True),
            #state size. (ngf*16) x 4 x 4
        )
        self.main = nn.Sequential(
            # nn.Linear(128, 4 * 4 * 16 * DIM),
            # nn.BatchNorm1d(4 * 4 * 16 * DIM),
            # nn.ReLU(True),
            # #state size. (ngf*16) x 4 x 4
            
            nn.ConvTranspose2d(16 * DIM, 8 * DIM, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(8 * DIM),
            nn.ReLU(True),
            #state size. (ngf*8) x 7 x 7

            nn.ConvTranspose2d(8 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * DIM),
            nn.ReLU(True),
            #state size. (ngf*4) x 14 x 14

            nn.ConvTranspose2d(4 * DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            #state size. (ngf*2) x 28 x 28
            
            nn.ConvTranspose2d(2 * DIM, DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            #state size. (ngf) x 56 x 56
            
            nn.ConvTranspose2d(DIM, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            #state size. (3) x 112 x 112
        )

        self.preprocess = preprocess

    def forward(self, z):
        DIM = self.DIM
        output = self.preprocess(z)
        # print(output.shape)
        output = output.view(-1, 16 * DIM, 4, 4)
        # print(output.shape)
        output = self.main(output)
        # print(output.shape)
        return output.view(-1, 3, 112, 112)


class Discriminator_112(nn.Module):
    def __init__(self, DIM=128):
        super(Discriminator_112, self).__init__()
        
        self.DIM = DIM
        
        main = nn.Sequential(
            #112x112
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #56x56
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #28x28
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #14x14
            nn.Conv2d(4 * DIM, 8 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #7x7
            nn.Conv2d(8 * DIM, 16 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #4x4
        )

        self.main = main
        self.linear = nn.Linear(16*4*4*DIM, 1)

    def forward(self, img):
        DIM = self.DIM
        
        output = self.main(img)
        # print(output.shape)
        output = output.view(-1, 16*4*4*DIM)
        output = self.linear(output)
        return output


def construct_generator(cfg, setup):
    if cfg.gan_type == "pretrain_biggan_imagenet256":
        from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                            save_as_images, display_in_terminal, convert_to_images)
        generator= BigGAN.from_pretrained('biggan-deep-256').to(setup['device'])
    elif cfg.gan_type == "dcgan_celeba32":
        generator = Generator_32().to(setup["device"])
        state_dict = torch.load(cfg.gan_state_dict_path)["state_dict"]
        generator.load_state_dict(state_dict)
    
    elif cfg.gan_type == "dcgan_celeba56" or cfg.gan_type == "dcgan_lfw56":
        generator = Generator_56().to(setup["device"])
        state_dict = torch.load(cfg.gan_state_dict_path)["state_dict"]
        generator.load_state_dict(state_dict)

    elif cfg.gan_type == "dcgan_celeba112" or cfg.gan_type == "dcgan_lfw112":
        generator = Generator_112().to(setup["device"])
        state_dict = torch.load(cfg.gan_state_dict_path)["state_dict"]
        generator.load_state_dict(state_dict)
    elif cfg.gan_type == "styleswin_celeba256":
        # from breaching.cases.models import styleswin_gan
        # # from styleswin_gan import Generator
        from .styleswin_gan import Generator
        args = {"size":256, "style_dim":512, "n_mlp":8, "channel_multiplier":2, "lr_mlp":0.01, "enable_full_resolution":8, "use_checkpoint":False}
        generator = Generator(**args).to(setup["device"])
        
        #load ckpt
        ckpt = torch.load(cfg.gan_state_dict_path, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(cfg.gan_state_dict_path)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass
        generator.load_state_dict(ckpt["g_ema"])
        
    elif cfg.gan_type == "styleswin_celeba1024":
        from .styleswin_gan import Generator
        args = {"size":1024, "style_dim":512, "n_mlp":8, "channel_multiplier":1, "lr_mlp":0.01, "enable_full_resolution":8, "use_checkpoint":False}
        generator = Generator(**args).to(setup["device"])
        
        #load ckpt
        ckpt = torch.load(cfg.gan_state_dict_path, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(cfg.gan_state_dict_path)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass
        generator.load_state_dict(ckpt["g_ema"])
    
    elif cfg.gan_type == "styleswin_bFFHQ256":
        from .styleswin_gan import Generator
        args = {"size":256, "style_dim":512, "n_mlp":8, "channel_multiplier":2, "lr_mlp":0.01, "enable_full_resolution":8, "use_checkpoint":False}
        generator = Generator(**args).to(setup["device"])
        
        #load ckpt
        ckpt = torch.load(cfg.gan_state_dict_path, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(cfg.gan_state_dict_path)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass
        generator.load_state_dict(ckpt["g_ema"])
    else:
        raise NotImplementedError(f"GAN mdoel {cfg.gan_type} not implement!")
    generator.eval()
    return generator

    