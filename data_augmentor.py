import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse


def append_hand(img, place_factor=0.7, rotate_factor=45, seed=23):
    if seed:
        np.random.seed(seed)

    aug_path = "images/augmentation/hands"
    aug_list = os.listdir(aug_path)
    i = np.random.randint(1, len(aug_list))
    aug = Image.open(os.path.join(aug_path, aug_list[i]))
    flip_factor = np.random.randint(0, 2)
    if flip_factor == 1:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
    scale_factor = img.width / aug.width * 0.7
    aug_h = int(aug.height * scale_factor)
    aug_w = int(aug.width * scale_factor)
    aug = aug.resize((aug_w, aug_h))

    x_min = img.width * (1 - place_factor) / 2 - aug_w / 2
    x_max = img.width * (1 + place_factor) / 2 - aug_w / 2
    x_base = np.random.randint(x_min, x_max)
    y_min = img.height * (1 - place_factor) / 2
    y_max = img.height * (1 + place_factor) / 2
    y_base = np.random.randint(y_min, y_max)
    angle = np.random.randint(-rotate_factor, rotate_factor)

    aug = aug.rotate(angle, expand=True)
    img.paste(aug, (x_base, y_base), aug)

    return img


def append_noise(img, rotate_factor=90, seed=23):
    if seed:
        np.random.seed(seed)

    aug_path = "images/augmentation/noise"
    aug_list = os.listdir(aug_path)
    i = np.random.randint(1, len(aug_list))
    aug = Image.open(os.path.join(aug_path, aug_list[i]))
    flip_factor = np.random.randint(0, 2)
    if flip_factor == 1:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
    aug = aug.resize((img.width, img.height))

    angle = np.random.randint(-rotate_factor, rotate_factor)
    aug = aug.rotate(angle, expand=True)

    img.paste(aug, (0, 0), aug)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", )
    args = parser.parse_args()

    face_image = Image.open(args.input)
    image_with_hand = append_hand(face_image, seed=False)
    noisy_image = append_noise(face_image, seed=False)

    plt.imshow(image_with_hand)
    plt.imshow(noisy_image)
    plt.show()
