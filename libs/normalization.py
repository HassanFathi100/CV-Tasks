"""Histogram normalization
    """
import numpy as np
from . import helper as Helper
from . import histogram as hist

from PIL import Image, ImageOps


def normalize_histogram(image_path: str):

    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_array = np.array(gray_image)

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
