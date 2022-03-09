"""Histogram normalization
    """
import numpy as np
from . import helper as Helper
from . import histogram as hist
import cv2


def normalize_histogram(image_path: str):

    img_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_array = np.array(img_grayscale)

    # get minimum and maximum pixel value in the image
    minimum_value = np.min(img_array)
    maximum_value = np.max(img_array)

    # normalize equation
    normalized_img = (img_array - minimum_value) * \
        (1.0 / (maximum_value - minimum_value))

    m, n = normalized_img.shape

    intensity_levels, count_intensity = hist.calculate_histogram(
        normalized_img)

    return (normalized_img, intensity_levels, count_intensity)
