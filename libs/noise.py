"""Functions that add noise to the image; Uniform, Gaussian and salt & pepper noise."""

import cv2
import numpy as np
import random
from . import helper as Helper


def gaussian_noise(image):
    mean = 0
    standard_deviation = 0.1
    noise = np.random.normal(mean, standard_deviation, image.shape)
    noise = noise.reshape(image.shape)
    noise = image + noise
    Helper.store_img('./output/gaussian_noise.jpg', noise)
    return noise


def uniform_noise(image):
    row, col, ch = image.shape
    noise = np.random.uniform(-255, 255, (row, col, ch))
    noise = noise.reshape(image.shape)
    noise = image + noise

    Helper.store_img('./output/uniform_noise.jpg', noise)
    return noise


def s_and_p_noise(image):  # be applied in greyscale image only

    # check if the image is colored or not
    if(len(image.shape) < 3):
        # print('gray')
        grayscale = image
    elif len(image.shape) == 3:
        # print('Color(RGB)')
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dimensions of the image
    row, col = grayscale.shape
    # salat mode
    # pick some pixels randomly to set them in white in range for example 200:20000
    number_of_pixels_salat = random.randint(200, 20000)
    for i in range(number_of_pixels_salat):
        # Pick a random y coordinate
        y_coordinate = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coordinate = random.randint(0, col - 1)

        # Color that pixel to white
        grayscale[y_coordinate][x_coordinate] = 255

    # peper mode
    number_of_pixels_pepper = random.randint(200, 20000)
    for i in range(number_of_pixels_pepper):
        # Pick a random y coordinate
        y_coordinate = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coordinate = random.randint(0, col - 1)

        # Color that pixel to black
        grayscale[y_coordinate][x_coordinate] = 0

    Helper.store_img('./output/s_and_p_noise.jpg', grayscale)
    return grayscale
