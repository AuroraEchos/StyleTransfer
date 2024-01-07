from PIL import Image
from torchvision import transforms

def gram_matrix(tensor):
    # 计算Gram矩阵的函数
    (b, ch, h, w) = tensor.size()
    features = tensor.view(b, ch, h*w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch*h*w)
    return gram

def load_image(filename, size=None, scale=None):
    # 加载图像的函数
    img = Image.open(filename).convert('RGB')
    if(size is not None):
        # 若指定了size，则调整图像大小
        img = img.resize((size, size), Image.ANTIALIAS)
    if(scale is not None):
        # 若指定了scale，则按比例调整图像大小
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    return img

def normalize_batch(batch):
    # 对图像进行标准化处理的函数，使用Imagenet的均值和标准差
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def match_size(image, imageToMatch):
    # 将输入的图像调整为与另一图像相同大小的函数
    img = image.clone().clamp(0, 255).numpy()
    img = img.squeeze(0)
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img = img.resize(
        (imageToMatch.shape[3], imageToMatch.shape[2]), Image.ANTIALIAS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img = transform(img)
    return img

def save_image(path, image):
    # 保存图像到指定路径的函数
    img = image.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(path)

def load_cam_image(img):
    # 从数组加载图像的函数
    img = Image.fromarray(img)
    return img

def show_cam_image(img):
    # 将图像转为NumPy数组并返回的函数
    img = img.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    return img
