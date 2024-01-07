from collections import namedtuple
import torch.nn as nn
from torchvision import models

# 定义一个VGG16模型的子类
class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        # 载入预训练的VGG16模型的特征提取部分
        vgg16_features = models.vgg16(pretrained=True).features

        # 定义切片（slices）
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # 将特征提取部分分割成4个子部分，按照论文中指定的层次
        for i in range(4):
            self.slice1.add_module(str(i), vgg16_features[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg16_features[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg16_features[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg16_features[i])

        # 如果requires_grad为False，则关闭所有参数的梯度计算
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 前向传播函数，返回各个切片的ReLU输出
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        # 使用namedtuple定义输出结构，包含四个切片的ReLU输出
        vgg_outputs = namedtuple(
            'VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
