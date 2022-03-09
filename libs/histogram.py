
import numpy as np
from . import helper as Helper


from PIL import Image, ImageOps


def calculate_histogram(gray_image: np.ndarray):
    """Calculate intensity values of all pixels in the image

    Args:
        img (np.ndarray): original image (grayscale)
        rows (int): number of rows in the image
        columns (int): number of columns in the image

    Returns:
        array: intensity values
        int: number of pixels for each intensity
    """

    # creating an og_image object
    # og_image = Image.open(image_path)
    # gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img = np.array(gray_image)

    rows, columns = img.shape

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
                if img[i, j] == intensity:
                    temp += 1
        count_intensity.append(temp)

    return (intensity_levels, count_intensity)
