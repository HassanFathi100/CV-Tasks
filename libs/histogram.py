
import numpy as np
from . import helper as Helper


def calculate_histogram(img_grayscale: np.ndarray, rows: int, columns: int):
    """Calculate intensity values of all pixels in the image

    Args:
        img_grayscale (np.ndarray): original image (grayscale)
        rows (int): number of rows in the image
        columns (int): number of columns in the image

    Returns:
        array: intensity values
        int: number of pixels for each intensity
    """

    # intensity values
    intensity_levels = []

    # number of pixels for each intensity
    count_intensity = []

    # calculate intensity values
    for intensity in range(0, 256):
        intensity_levels.append(intensity)
        temp = 0

        # loops on each pixel
        for i in range(rows):
            for j in range(columns):
                if img_grayscale[i, j] == intensity:
                    temp += 1
        count_intensity.append(temp)

    Helper.plot_histogram(intensity_levels, count_intensity)
    return (intensity_levels, count_intensity)
