"""Implement here functions that filter the noisy images using the following low pass filters:
    Average, Gaussian and median filters.
    """

import numpy as np
import numpy 
from PIL import Image
<<<<<<< Updated upstream
import helper as Helper
import cv2
=======
from . import helper as Helper


from PIL import Image, ImageOps
>>>>>>> Stashed changes


def average_filter(image_path: str):
    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_grayscale = np.array(gray_image)

    m, n = img_grayscale.shape

    # Develop Averaging filter(3, 3) mask
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9

    # Convolving the 3X3 mask over the input image
    img_new = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img_grayscale[i-1, j-1]*mask[0, 0]+img_grayscale[i-1, j]*mask[0, 1]+img_grayscale[i-1, j + 1]*mask[0, 2]+img_grayscale[i, j-1]*mask[1, 0] + img_grayscale[i, j] * \
                mask[1, 1]+img_grayscale[i, j + 1]*mask[1, 2]+img_grayscale[i + 1, j-1]*mask[2,
                                                                                             0]+img_grayscale[i + 1, j]*mask[2, 1]+img_grayscale[i + 1, j + 1]*mask[2, 2]

            img_new[i, j] = temp

    average_filterd_img = img_new.astype(np.uint8)

    Helper.store_img_cv2(
        './output/average_filterd_img.jpg', average_filterd_img)
    return average_filterd_img


# def median_filter_calc(data, filter_size):
#     temp = []
#     indexer = filter_size // 2
#     data_final = []
#     data_final = np.zeros((len(data), len(data[0])))
#     for i in range(len(data)):

#         for j in range(len(data[0])):

#             for z in range(filter_size):
#                 if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
#                     for c in range(filter_size):
#                         temp.append(0)
#                 else:
#                     if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
#                         temp.append(0)
#                     else:
#                         for k in range(filter_size):
#                             temp.append(data[i + z - indexer][j + k - indexer])

#             temp.sort()
#             data_final[i][j] = temp[len(temp) // 2]
#             temp = []

#     return data_final


# def median_filter(img_path):
#     img = Image.open(img_path).convert("L")
#     arr = np.array(img)
#     noise_free_arr = median_filter_calc(arr, 6)
#     noise_free_img = Image.fromarray(noise_free_arr)
#     noise_free_img.show()
#     Helper.store_img_cv2('./output/noise_free_img.jpg', noise_free_img)

def Median_Filter_Calculation(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
            
    return data_final


def Median_filter():
    img = Image.open("assets/lion.jpg").convert("L")
    arr = numpy.array(img)
    removed_noise = Median_Filter_Calculation(arr, 6) 
    img = Image.fromarray(removed_noise)
    img.show()
    Helper.store_img_cv2('./output/noise_free_img.jpg', img)


def correlation(img_grayscale, mask):
    row, col = img_grayscale.shape
    m, n = mask.shape
    new = np.zeros((row+m-1, col+n-1))
    n = n//2
    m = m//2
    filtered_img = np.zeros(img_grayscale.shape)
    new[m:new.shape[0]-m, n:new.shape[1]-n] = img_grayscale
    for i in range(m, new.shape[0]-m):
        for j in range(n, new.shape[1]-n):
            temp = new[i-m:i+m+1, j-m:j+m+1]
            result = temp*mask
            filtered_img[i-m, j-n] = result.sum()

    return filtered_img


def gaussian_filter(m, n, sigma):
    gaussian = np.zeros((m, n))
    m = m//2
    n = n//2
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m, y+n] = (1/x1)*x2

    return gaussian


def apply_gaussian_filter(image_path: str):

    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_grayscale = np.array(gray_image)

    g = gaussian_filter(5, 5, 2)
    n = correlation(img_grayscale, g)

    gaussian_filtered_img = n.astype(np.uint8)

    Helper.store_img_cv2('./output/gaussian_filtered_img.jpg',
                         gaussian_filtered_img)
    return gaussian_filtered_img


Median_filter ()
# AverageFilter()
