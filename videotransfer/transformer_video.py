import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm='instance'):
        """
        Convolutional层定义，包含反射填充和标准化。

        Params:
        - in_channels: 输入图像的通道数
        - out_channels: 卷积输出的通道数
        - kernel_size: 卷积核的大小
        - stride: 卷积的步长
        - norm: 标准化类型，默认为 'instance'。可选值：['instance', 'batch', 'None']
        """
        super(ConvLayer, self).__init__()
        # 添加反射填充
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # 卷积层
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

        # 标准化层
        self.norm_type = norm
        if norm == 'instance':
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        assert norm in ['instance', 'batch', 'None'], 'Accepted values must belong to: "instance", "batch", "None"'

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == 'None':
            out = x
        else:
            out = self.norm_layer(x)
        return out


class ResidualLayer(nn.Module):
    """
    保留连接的残差块。
    """
    def __init__(self, channels=128, kernel_size=3):
        """
        Params:
        - channels: 通道数，默认为128
        - kernel_size: 卷积核的大小，默认为3
        """
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, 1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, 1)

    def forward(self, x):
        # 保存残差
        residue = x
        # 第1层输出 + 激活函数
        out = self.relu(self.conv1(x))
        # 第2层输出
        out = self.conv2(x)
        # 将残差加到该输出上
        out = out + residue
        return out


class DeConvLayer(nn.Module):
    """
    转置卷积层或分数步长卷积层。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        """
        Params:
        - in_channels: 输入图像的通道数
        - out_channels: 卷积输出的通道数
        - kernel_size: 卷积核的大小
        - stride: 卷积的步长
        - output_padding: 输出形状的一侧添加的额外大小
        - norm: 标准化类型，默认为 'instance'。可选值：['instance', 'batch', 'None']
        """
        super(DeConvLayer, self).__init__()

        # 转置卷积或分数步长卷积
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # 标准化层
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        assert norm in ['instance', 'batch', 'None'], 'Accepted values must belong to: "instance", "batch", "None"'

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == 'None':
            out = x
        else:
            out = self.norm_layer(x)
        return out


class TransformNet(nn.Module):
    """
    与Johnson等人论文中描述的图像转换网络相对应
    论文链接: https://arxiv.org/abs/1603.08155
    """
    def __init__(self):
        """
        Conv块 -> Residual块 -> DeConv块
        """
        super(TransformNet, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.DeConvBlock = nn.Sequential(
            DeConvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeConvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm='None')
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeConvBlock(x)
        return out
