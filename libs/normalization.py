"""Histogram normalization
    """
import numpy as np
from . import helper as Helper
from . import histogram as hist


def normalize_histogram(img_grayscale: np.ndarray):
    img_array = np.array(img_grayscale)

    # get minimum and maximum pixel value in the image
    minimum_value = np.min(img_array)
    maximum_value = np.max(img_array)

    # normalize equation
    normalized_img = (img_array - minimum_value) * \
        (1.0 / (maximum_value - minimum_value))
    m, n = normalized_img.shape

    intensity_levels, count_intensity = hist.calculate_histogram(
        normalized_img, m, n)

    Helper.plot_normalized(intensity_levels, count_intensity)

    return (normalized_img, intensity_levels, count_intensity)
