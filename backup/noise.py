"""Functions that add noise to the image; Uniform, Gaussian and salt & pepper noise."""

import numpy as np
import random
from . import helper as Helper
import matplotlib.pyplot as plt


from PIL import Image, ImageOps


def gaussian_noise(image_grayscale: np.ndarray):

    # Copy the original image
    image = np.copy(image_grayscale)

    mean = 0
    standard_deviation = 0.1
    noise = np.random.normal(mean, standard_deviation, image.shape)
    noise = noise.reshape(image.shape)
    noise = image + noise
    noise = noise * 255
    Helper.store_img_pil('./output/gaussian_noise.jpg', noise)
    return noise


def uniform_noise(image_grayscale: np.ndarray):

    # Copy the original image
    image = np.copy(image_grayscale)

    row, col, ch = image.shape
    noise = np.random.uniform(-255, 255, (row, col, ch))
    noise = noise.reshape(image.shape)
    noise = image + noise

    Helper.store_img_pil('./output/uniform_noise.jpg', noise)
    return noise


def s_and_p_noise(image: np.ndarray):  # be applied in greyscale image only

    # Copy the original image
    img_grayscale = np.copy(image)

    # dimensions of the image
    row, col = img_grayscale.shape
    # salat mode
    # pick some pixels randomly to set them in white in range for example 200:20000
    number_of_pixels_salat = random.randint(200, 20000)
    for i in range(number_of_pixels_salat):
        # Pick a random y coordinate
        y_coordinate = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coordinate = random.randint(0, col - 1)

        # Color that pixel to white
        img_grayscale[y_coordinate][x_coordinate] = 255

    # peper mode
    number_of_pixels_pepper = random.randint(200, 20000)
    for i in range(number_of_pixels_pepper):
        # Pick a random y coordinate
        y_coordinate = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coordinate = random.randint(0, col - 1)

        # Color that pixel to black
        img_grayscale[y_coordinate][x_coordinate] = 0

    Helper.store_img_pil('./output/s_and_p_noise.jpg', img_grayscale)
    return img_grayscale
