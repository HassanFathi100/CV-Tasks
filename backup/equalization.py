"""Histogram equalization"""

import numpy as np
from matplotlib import pyplot as plt
from . import helper as Helper
from . import histogram


from PIL import Image, ImageOps


def equalization(image_path: str):

    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_grayscale = np.array(gray_image)

    rows, columns = img_grayscale.shape

    # Histogram
    _, intensity_count = histogram.calculate_histogram(img_grayscale)

    # Converting list to numpay array
    intensity_count_array = np.array(intensity_count)

    # Get the sum of all bins
    intensity_count_sum = np.sum(intensity_count_array)

    # Calculating propability density function
    PDF = intensity_count_array/intensity_count_sum

    # Calculating Cumulative density function
    CDF = np.array([])
    CDF = np.cumsum(PDF)

    # Rounding CDF values
    equalized_histogram = np.round((255 * CDF), decimals=0)

    # Flattening the image
    img_vector = img_grayscale.ravel()

    # Converting 1D array (vector) to 2D array (image)
    mapped_img_vector = []
    for pixel in img_vector:
        mapped_img_vector.append(equalized_histogram[pixel])

    equalized_img = np.reshape(np.asarray(mapped_img_vector),
                               img_grayscale.shape).astype(np.uint8)

    Helper.store_img_pil('./output/equalized_img.jpg', equalized_img)
    return equalized_img

# Already defined equalization function
# equalizedImg_alreadyDefine = cv2.equalizeHist(img)


# Already defined histogram function
# plt.hist(img.ravel(),256,[0,256]); plt.show()
