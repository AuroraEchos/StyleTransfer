# Importing the resources
import argparse
import sys
import torch
from torchvision import transforms
from transformer_video import TransformNet
from utils_video import load_image, match_size, save_image

def stylize(args):
    # 选择可用的GPU
    device = torch.device('cuda' if args.cuda else 'cpu')

    # 加载内容图像
    content_image = load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    # 设置requires_grad为False
    with torch.no_grad():
        style_model = TransformNet()
        state_dict = torch.load(args.model)

        # 加载模型的学到的参数
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        # 输出图像
        output = style_model(content_image).cpu()

    content_image = match_size(content_image, output)
    weighted_output = output * args.style_strength + \
        (content_image * (1 - args.style_strength))
    save_image(args.output_image, weighted_output[0])

# 命令行参数解析
eval_arg_parser = argparse.ArgumentParser(
    description="parser for fast-neural-style-evaluation")
eval_arg_parser.add_argument("--content-image", type=str, required=True,
                             help="要进行风格化的内容图像的路径")
eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                             help="缩小内容图像的因子")
eval_arg_parser.add_argument("--output-image", type=str, required=True,
                             help="保存输出图像的路径")
eval_arg_parser.add_argument("--model", type=str, required=True,
                             help="用于风格化图像的保存模型")
eval_arg_parser.add_argument("--cuda", type=int, required=True,
                             help="将其设置为1以在GPU上运行，设置为0以在CPU上运行")
eval_arg_parser.add_argument("--style-strength", type=float, default=1.0,
                             help="在0和1之间设置风格的强度，默认为1.0")

args = eval_arg_parser.parse_args()

if(args.cuda and not torch.cuda.is_available()):
    print('ERROR: cuda不可用，请尝试在CPU上运行')
    sys.exit(1)

stylize(args)
