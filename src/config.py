import torch

STYLE_WEIGHTS = {
    "conv1_1": 1.,
    "conv2_1": 0.75,
    "conv3_1": 0.5,
    "conv4_1": 0.25,
    "conv5_1": 0.1
}

CONTENT_WEIGHT = 1  # Alpha

STYLE_WEIGHT = 1e4  # Beta

STEPS = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTENT_IMAGE = "tiger.jpg"

STYLE_IMAGE = "wave.jpg"

LEARNING_RATE = 0.004
