import numpy as np


def frequency_domain_fun(image):
    """
    padding function to make size of mask the same of image to keep the size 
    of image the same as before applying filtering
    """
    grayscale_image = np.copy(image)

    # getting fft image
    fft_image = np.fft.fft2(grayscale_image)
    shfft_image = np.fft.fftshift(fft_image)
    return shfft_image


def padding_fun(image, width, height, padding_value):
    img_width, img_height = image.shape
    right_width_added = (width-img_width)//2
    left_width_added = width-(right_width_added+img_width)
    upper_height_added = (height-img_height)//2
    lower_height_added = height-(img_height+upper_height_added)
    padding_img = np.pad(image, ((right_width_added, left_width_added), (upper_height_added, lower_height_added)),
                         constant_values=padding_value)
    return padding_img


"""
filter_fun is a function take arguments : original image , mask size which applied in frequency_domain_image
and filter_type as a string to select low or high pass filter 
in low pass filter case :making zero mask of specific size then getting pad with constant value "1"
with the same size of original image
in high pass filter case :making ones mask of specific size then getting pad with constant value "0"
with the same size of original image

"""


def filter_fun(image: np.ndarray, size, filter_type):
    image_grayscale = np.copy(image)
    row, col = image_grayscale.shape
    frequency_domain_image = frequency_domain_fun(image_grayscale)
    if filter_type == "lpf":
        mask = np.ones((size, size))
        padding_mask = padding_fun(mask, row, col, 0)
    elif filter_type == "hpf":
        mask = 1-np.ones((size, size))
        padding_mask = padding_fun(mask, row, col, 1)

    filtered_image = np.fft.ifftshift(frequency_domain_image*padding_mask)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image).clip(0, 255).astype(np.uint8)


    return filtered_image

