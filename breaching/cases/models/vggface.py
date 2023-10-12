
import torch.nn as nn
from collections import OrderedDict


class VGGFace(nn.Module):
    def __init__(self, num_classes=None):
        super(VGGFace, self).__init__()

        self.features = nn.ModuleDict(OrderedDict(
            {
                # === Block 1 ===
                'conv_1_1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                'relu_1_1': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                'relu_1_2': nn.ReLU(inplace=True),
                'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2), # shape // 2
                # === Block 2 ===
                'conv_2_1': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                'relu_2_1': nn.ReLU(inplace=True),
                'conv_2_2': nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                'relu_2_2': nn.ReLU(inplace=True),
                'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2), # shape // 2
                # === Block 3 ===
                'conv_3_1': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                'relu_3_1': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_2': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_3': nn.ReLU(inplace=True),
                'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), # shape // 2
                # === Block 4 ===
                'conv_4_1': nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                'relu_4_1': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_2': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_3': nn.ReLU(inplace=True),
                'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2), # shape // 2
                # === Block 5 ===
                'conv_5_1': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_1': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_2': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_3': nn.ReLU(inplace=True),
                # 'maxp_5_3': nn.MaxPool2d(kernel_size=2, stride=2) # shape // 2
                'maxp_5_3': nn.AdaptiveAvgPool2d((1,1)) # shape // 2
            }))
        
        # self.fc = nn.ModuleDict(OrderedDict(
        #     {
        #         'fc6': nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        #         'fc6-relu': nn.ReLU(inplace=True),
        #         'fc6-dropout': nn.Dropout(p=0.5),
        #         'fc7': nn.Linear(in_features=4096, out_features=4096),
        #         'fc7-relu': nn.ReLU(inplace=True),
        #         'fc7-dropout': nn.Dropout(p=0.5),
        #         'fc8': nn.Linear(in_features=4096, out_features=2622),
        #     }))
        self.fc = nn.ModuleDict(OrderedDict(
            {
                "fc6": nn.Linear(in_features=512, out_features=num_classes)
            }
        ))
        
        

    def forward(self, x):
        # Forward through feature layers
        for k, layer in self.features.items():
            x = layer(x)
            # print(k, x.shape)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        for k, layer in self.fc.items():
            x = layer(x)

        return x