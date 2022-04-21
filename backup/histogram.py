import numpy as np


def calculate_histogram(gray_image: np.ndarray):
    """Calculate intensity values of all pixels in the image

    Args:
        gray_image (np.ndarray): original image (grayscale)

    Returns:
        array: intensity values
        int: number of pixels for each intensity
    """
    interpolate_data = np.round(np.interp(
        gray_image, (gray_image.min(), gray_image.max()), (0, 255))).astype('uint8')

    intensity_levels = np.arange(0, 256)
    count_intensity = np.bincount(interpolate_data.ravel(), minlength=255)

    return (intensity_levels, count_intensity)
