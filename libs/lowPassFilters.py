"""Implement here functions that filter the noisy images using the following low pass filters:
    Average, Gaussian and median filters.
    """

from libs import helper as Helper
import numpy as np
import numpy
from PIL import Image


from PIL import Image, ImageOps


def average_filter(image_path: str):
    """ performing an average filter

    Args:
        image_path (str): the input image 

    Returns:
        average_filterd_img (nd array): the output image 
    """
    # creating an a gray scale image
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to a numpy array
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

    Helper.store_img_pil(
        './output/average_filterd_img.jpg', average_filterd_img)
    return average_filterd_img


def Median_Filter_Calculation(data, filter_size):
    """ performing a median filter 

    Args:
        data (array): the input photo
        filter_size (): size of mask

    Returns:
        data_final(nd array): the output photo
    """
    # reseting the temporary list to zero
    temp = []
    # setting the boundries of the filter
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data), len(data[0])))
    # Looping over the image indices
    for i in range(len(data)):

        for j in range(len(data[0])):

            # Looping over the filter indices

            for z in range(filter_size):
                # checking if the rows and columns are going out of bounds with these if statements
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            # where i = image row,z = filter row, j = image column, k  = filter coloumn
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]

            # reseting the temporary list to zero
            temp = []

    return data_final


def median_filter(image_path: str):
    """ a function that simply calls the Median_Filter_Calculation function and does some conversions  
    """
    img = Image.open(image_path).convert("L")
    # coverts image into array
    arr = numpy.array(img)
    # calls Median_Filter_Calculation function and pass the array (image) and filter size
    removed_noise = Median_Filter_Calculation(arr, 6)
    # convert from array to image
    img = Image.fromarray(removed_noise)
    Median_filtered_img = removed_noise.astype(np.uint8)
    Helper.store_img_pil(
        './output/median_filtered_img.jpg', Median_filtered_img)
    return Median_filtered_img


def correlation(img_grayscale, mask):
    """a function that performs the convolution of the gaussian filter + the input image 

    Args:
        img_grayscale (array): the input image 
        mask (array): the mask(kernel)

    Returns:
        filtered_img
    """
    row, col = img_grayscale.shape
    m, n = mask.shape
    new = np.zeros((row+m-1, col+n-1))
    # setting the boundries of the image array
    n = n//2
    m = m//2
    filtered_img = np.zeros(img_grayscale.shape)
    new[m:new.shape[0]-m, n:new.shape[1]-n] = img_grayscale
    # looping over the image row indices
    for i in range(m, new.shape[0]-m):

        # looping over the image coloumn indices
        for j in range(n, new.shape[1]-n):
            temp = new[i-m:i+m+1, j-m:j+m+1]
            result = temp*mask
            filtered_img[i-m, j-n] = result.sum()

    return filtered_img


def gaussian_filter(m, n, sigma):
    """_summary_

    Args:
        m : rows
        n : columns
        sigma: the standard deviation 

    Returns:
        gaussian: the filter array
    """
    # empty array
    gaussian = np.zeros((m, n))
    # setting the boundries of the filter
    m = m//2
    n = n//2
    # looping over rows
    for x in range(-m, m+1):

        # looping over rows
        for y in range(-n, n+1):
            # applying the equation of gaussian
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m, y+n] = (1/x1)*x2

    return gaussian


def apply_gaussian_filter(image_path: str):
    """ function that calls the gaussian_filter and correlation functions and does some conversions

    Args:
        image_path (str): input image

    Returns:
        gaussian_filtered_img:
    """
    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_grayscale = np.array(gray_image)
    # img_grayscale = np.array(og_image)

    g = gaussian_filter(7, 7, 7)
    n = correlation(img_grayscale, g)

    gaussian_filtered_img = n.astype(np.uint8)

    Helper.store_img_pil('./output/gaussian_filtered_img.jpg',
                         gaussian_filtered_img)
    return gaussian_filtered_img
