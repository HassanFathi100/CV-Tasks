import numpy as np
import math
from .frequency_domain_filters import filter_fun
from .helper import store_img_pil


def square_image(img: np.ndarray):
    """_make the image square (height = width)_

    Args:
        img (np.ndarray): _the image to square_

    Return:
        the image with black square fit
    """

    height, width = img.shape

    if(height > width):
        paddingValue = math.floor((height - width)/2)
        paddedImage = np.pad(
            img, [(paddingValue, paddingValue), (0, 0)], 'constant', constant_values=(0))
    elif(width > height):
        paddingValue = math.floor((width - height)/2)
        paddedImage = np.pad(
            img, [(paddingValue, paddingValue), (0, 0)], 'constant', constant_values=(0))
    else:
        paddedImage = img

    return paddedImage


def hybrid_image(img1_grayscale: np.ndarray, img2_grayscale: np.ndarray):
    """
    _Calculate the hybrid image_

    Args:
        img1 (np.ndarray): _high frequency image_
        img2 (np.ndarray): _low frequency image_

    Return:
        hybrid image array
    """

    highFrequencyImg = filter_fun(img1_grayscale, 25, "hpf")
    lowFrequencyImg = filter_fun(img2_grayscale, 25, "lpf")

    hybridImg = highFrequencyImg + lowFrequencyImg
    store_img_pil("./output/hybridImg.jpg", hybridImg.astype(np.uint8))
    return hybridImg
