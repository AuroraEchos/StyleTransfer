import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from imagetransfer.utils_image import *
from imagetransfer.models_image import *
import random

import warnings
warnings.filterwarnings("ignore")

# 1. 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, 
    upsample=None, instance_norm=True, relu=True):
    layers = []
    if upsample:
        layers.append(torch.nn.Upsample(mode='nearest', scale_factor=upsample))
    layers.append(torch.nn.ReflectionPad2d(kernel_size // 2))
    layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(torch.nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(torch.nn.ReLU())
    return layers

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1), 
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x

class TransformNet(torch.nn.Module):
    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.downsampling = torch.nn.Sequential(
            *ConvLayer(3, base, kernel_size=9), 
            *ConvLayer(base, base*2, kernel_size=3, stride=2), 
            *ConvLayer(base*2, base*4, kernel_size=3, stride=2), 
        )
        self.residuals = torch.nn.Sequential(*[ResidualBlock(base*4) for i in range(5)])
        self.upsampling = torch.nn.Sequential(
            *ConvLayer(base*4, base*2, kernel_size=3, upsample=2),
            *ConvLayer(base*2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False),
        )
    
    def forward(self, X):
        y = self.downsampling(X)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y
    
""" 
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256), 
    torchvision.transforms.CenterCrop(256), 
    torchvision.transforms.ToTensor(), 
    tensor_normalizer, 
])

# 2. 加载风格图像
style_path = "images/style_images/mosaic.jpg"
style_img = read_image(style_path).to(device)

# 3. 加载预训练的模型参数
transform_net = TransformNet(32).to(device)
transform_net.load_state_dict(torch.load('imagetransfer/pth/transform_mosaic.pth'))

# 4. 将模型设置为评估模式
transform_net.eval()

# 5. 随机选择一张图片进行推理
dataset = torchvision.datasets.ImageFolder('E:\\coco', transform=data_transform)
content_img = random.choice(dataset)[0].unsqueeze(0).to(device)
print(content_img)
output_img = transform_net(content_img)

# 6. 展示结果
plt.figure(figsize=(18, 12))

plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')

plt.subplot(1, 3, 2)
imshow(content_img, title='Content Image')

plt.subplot(1, 3, 3)
imshow(output_img.detach(), title='Output Image')

plt.show() """