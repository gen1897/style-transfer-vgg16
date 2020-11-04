import torch
import torch.nn as nn
import torchvision.models as models


pools = [4, 9, 18, 27, 36]


def set_net(DEVICE: str):
    vgg = models.vgg19(pretrained=True).features

    # Freeze parameters
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)

    vgg.to(DEVICE)

    for pool in pools:
        vgg[pool] = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=False)

    return vgg
