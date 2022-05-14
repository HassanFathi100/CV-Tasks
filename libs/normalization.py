"""Histogram normalization
    """
import numpy as np
# from . import histogram as hist


def normalize_histogram(gray_image: np.ndarray):

    # Copy img array
    img_array = np.copy(gray_image)

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
