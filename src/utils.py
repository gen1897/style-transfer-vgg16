# Libraries
from skimage import io

import numpy as np
from skimage import img_as_ubyte

from PIL import Image
import torch
from torchvision import transforms


# Load image
def load_image(image_path: str, max_size=400, shape=None):

    # Skimage allows python to read images on local or internet with the same funciton
    image = io.imread(image_path)
    image = Image.fromarray(image)

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    image_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    return image_transforms(image)[:3, :, :].unsqueeze(0)


# Convert image to numpy
def im_convert(image: torch.float32):
    image = image.to("cpu").clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image: torch.float32, model):

    layers = {
        # Style
        '0': 'conv1_1',
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "28": "conv5_1",
        # Content
        "7": "conv2_2",
        "12": "conv3_2",
        "21": "conv4_2",
        "30": "conv5_2",
        # Pools
        "4": "pool_1",
        "9": "pool_2",
        "18": "pool_3",
        "27": "pool_4",
        "36": "pool_5"
    }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    # Get tensor size
    _, d, h, w = tensor.size()

    # Reshape for features in each channel
    tensor = tensor.view(d, h * w)

    # Gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram
