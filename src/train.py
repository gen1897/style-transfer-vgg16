import utils
import model
from config import *

import torch.optim as optim
import matplotlib.pyplot as plt


def main():
    # Load Model
    MODEL = model.set_net(DEVICE)

    print("Loading images...")
    content = utils.load_image("input/{}".format(CONTENT_IMAGE)).to(DEVICE)
    style = utils.load_image(
        "input/{}".format(STYLE_IMAGE), shape=content.shape[-2:]).to(DEVICE)

    # Get features
    content_features = utils.get_features(content, MODEL)
    style_features = utils.get_features(style, MODEL)

    # Get Gramian matrix
    style_grams = {layer: utils.gram_matrix(
        style_features[layer]) for layer in style_features}

    # Create a target image or canvas to draw content with another style
    target = content.clone().requires_grad_(True).to(DEVICE)

    # Show image every X steps
    show_every = int(STEPS / 5)

    # L-BFGS gave problems, so I used ADAM
    optimizer = optim.Adam([target], lr=LEARNING_RATE)

    # ======== Transfer process ========

    for step in range(1, STEPS+1):
        target_features = utils.get_features(target, MODEL)

        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0

        for layer in STYLE_WEIGHTS:
            # Target style representation
            target_feature = target_features[layer]
            target_gram = utils.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # Style style representation
            style_gram = style_grams[layer]
            # Style loss for the layer
            layer_loss = STYLE_WEIGHTS[layer] * \
                torch.mean((target_gram - style_gram)**2)

            style_loss += layer_loss / (d*h*w)

        # Total loss
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % show_every == 0:
            print("Total loss: {}".format(total_loss.item()))

    # display content and final, target image
    plt.imsave("output/styled-{}".format(CONTENT_IMAGE),
               utils.im_convert(target))


if __name__ == "__main__":

    main()
