import torch
import torchvision
from PIL import Image
import numpy as np
from tqdm import tqdm


from utils_image import *
from models_image import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

style_path = "images/style_images/picasso.jpg"
style_img = read_image(style_path).to(device)
imshow(style_img, title='Style Image')

weights_path = 'imagetransfer/pth/vgg16.pth'
pretrained_vgg16 = torchvision.models.vgg16(pretrained=False)
pretrained_vgg16.load_state_dict(torch.load(weights_path))
vgg16 = VGG(pretrained_vgg16.features[:23]).to(device).eval()

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
    
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

batch_size = 4
width = 256

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(width), 
    torchvision.transforms.CenterCrop(width), 
    torchvision.transforms.ToTensor(), 
    tensor_normalizer, 
])

dataset = torchvision.datasets.ImageFolder('E:\\coco', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#print(dataset)

style_features = vgg16(style_img)
style_grams = [gram_matrix(x) for x in style_features]
style_grams = [x.detach() for x in style_grams]

def tensor_to_array(tensor):
    x = tensor.cpu().detach().numpy()
    x = (x*255).clip(0, 255).transpose(0, 2, 3, 1).astype(np.uint8)
    return x

def save_debug_image(style_images, content_images, transformed_images, filename):
    style_image = Image.fromarray(recover_image(style_images))
    content_images = [recover_image(x) for x in content_images]
    transformed_images = [recover_image(x) for x in transformed_images]
    
    new_im = Image.new('RGB', (style_image.size[0] + (width + 5) * 4, max(style_image.size[1], width*2 + 5)))
    new_im.paste(style_image, (0,0))
    
    x = style_image.size[0] + 5
    for i, (a, b) in enumerate(zip(content_images, transformed_images)):
        new_im.paste(Image.fromarray(a), (x + (width + 5) * i, 0))
        new_im.paste(Image.fromarray(b), (x + (width + 5) * i, width + 5))
    
    new_im.save(filename)

transform_net = TransformNet(32).to(device)


verbose_batch = 800
style_weight = 1e5
content_weight = 1
tv_weight = 1e-6

optimizer = torch.optim.Adam(transform_net.parameters(), 1e-3)
transform_net.train()

n_batch = len(data_loader)

for epoch in range(1):
    print('Epoch: {}'.format(epoch+1))
    smooth_content_loss = Smooth()
    smooth_style_loss = Smooth()
    smooth_tv_loss = Smooth()
    smooth_loss = Smooth()
    with tqdm(enumerate(data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            optimizer.zero_grad()

            # 使用风格模型预测风格迁移图像
            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)
            transformed_images = transformed_images.clamp(-3, 3)

            # 使用 vgg16 计算特征
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)

            # content loss
            content_loss = content_weight * torch.nn.functional.mse_loss(transformed_features[1], content_features[1])
            
            # total variation loss
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            # style loss
            style_loss = 0.
            transformed_grams = [gram_matrix(x) for x in transformed_features]
            for transformed_gram, style_gram in zip(transformed_grams, style_grams):
                style_loss += style_weight * torch.nn.functional.mse_loss(transformed_gram, 
                                                        style_gram.expand_as(transformed_gram))

            # 加起来
            loss = style_loss + content_loss + tv_loss

            loss.backward()
            optimizer.step()

            smooth_content_loss += content_loss.item()
            smooth_style_loss += style_loss.item()
            smooth_tv_loss += tv_loss.item()
            smooth_loss += loss.item()
            
            s = f'Content: {smooth_content_loss:.2f} '
            s += f'Style: {smooth_style_loss:.2f} '
            s += f'TV: {smooth_tv_loss:.4f} '
            s += f'Loss: {smooth_loss:.2f}'
            if batch % verbose_batch == 0:
                s = '\n' + s
                save_debug_image(style_img, content_images, transformed_images, 
                                 f"imagetransfer/debug/s2_{epoch}_{batch}.jpg")
            
            pbar.set_description(s)
    torch.save(transform_net.state_dict(), 'imagetransfer/pth/transform_picasso.pth')