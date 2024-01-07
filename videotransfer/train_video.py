import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformer_video import TransformNet
from utils_video import gram_matrix, load_image, normalize_batch
from vgg_video import VGG16

import warnings
warnings.filterwarnings("ignore")

# 检查文件夹是否存在，如果不存在则创建
def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not os.path.exists(args.checkpoint_model_dir):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

# 训练过程
def train(args):
    # 选择可用的GPU
    device = torch.device('cuda' if args.cuda else 'cpu')

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # 数据集和数据加载器
    train_dataset = datasets.ImageFolder(args.dataset, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 加载网络模型
    transformer = TransformNet().to(device)
    vgg = VGG16(False).to(device)

    # 优化器
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)

    # 损失函数
    mse_loss = nn.MSELoss()

    # 风格特征
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(x) for x in features_style]

    # 训练循环
    for epoch in range(args.epochs):
        transformer.train()
        total_content_loss = 0.0
        total_style_loss = 0.0
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):

            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            # 从变换网络中得到输出 -> y
            x = x.to(device)
            y = transformer(x)

            # 标准化批次 (y-> 变换网络的输出, x-> 原始输入)
            y = normalize_batch(y)
            x = normalize_batch(x)

            # 从vgg模型中得到输出
            features_y = vgg(y)
            features_x = vgg(x)

            # 计算内容损失
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # 计算风格损失
            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            # 计算批次损失
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # 计算总损失
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                print(f'{time.ctime()}\tEpoch {epoch+1}:\t[{count}/{len(train_dataset)}]\tcontent: {total_content_loss / batch_id + 1}\tstyle: {total_style_loss / batch_id + 1}\ttotal: {(total_content_loss + total_style_loss) / (batch_id + 1)}')

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = 'ckpt_epoch_' + str(epoch) + "_batch_id_" + str(batch_id + 1) + '.pth'
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device)
    
    # 保存模型
    transformer.eval().cpu()
    save_model_filename = 'epoch_' + str(args.epochs) + '_' + str(time.ctime()).replace(' ', '_') + '_' + str(args.content_weight) + '_' + str(args.style_weight) + '.model'
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print('\n模型训练完成!已保存到:', save_model_path)

# 命令行参数
arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style-training")

arg_parser.add_argument("--epochs", type=int, default=2,
                        help="训练的轮数，默认为2")
arg_parser.add_argument("--batch-size", type=int, default=4,
                        help="训练的批次大小，默认为4")
arg_parser.add_argument("--dataset", type=str, default="E://coco",
                        help="训练数据集的路径，该路径应指向包含所有训练图像的文件夹")
arg_parser.add_argument("--style-image", type=str, default="videotransfer/images/style_images/mosaic.jpg",
                        help="风格图像的路径")
arg_parser.add_argument("--save-model-dir", type=str, default="videotransfer/models",
                        help="保存训练模型的文件夹路径")
arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                        help="保存已训练模型检查点的文件夹路径")
arg_parser.add_argument("--image-size", type=int, default=256,
                        help="训练图像的大小，默认为256 X 256")
arg_parser.add_argument("--style-size", type=int, default=None,
                        help="风格图像的大小，默认为风格图像的原始大小")
arg_parser.add_argument("--cuda", type=int, default=1,
                        help="设置为1在GPU上运行，0在CPU上运行")
arg_parser.add_argument("--seed", type=int, default=42,
                        help="训练的随机种子")
arg_parser.add_argument("--content-weight", type=float, default=1e5,
                        help="内容损失的权重，默认为1e5")
arg_parser.add_argument("--style-weight", type=float, default=1e10,
                        help="风格损失的权重，默认为1e10")
arg_parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率，默认为1e-3")
arg_parser.add_argument("--log-interval", type=int, default=500,
                        help="记录训练损失的图像数量， 默认为500")
arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                        help="在训练模型的每个多少个批次后创建已训练模型的检查点")

args = arg_parser.parse_args()

if args.cuda and not torch.cuda.is_available():
    print("错误：cuda不可用，请尝试在CPU上运行")
    sys.exit(1)

check_paths(args)
train(args)
