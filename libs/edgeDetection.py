"""Detect edges using the following masks: Sobel, Roberts, Prewitt and Canny edge detectors."""


from . import helper as Helper
import numpy as np
from . import lowPassFilters
from scipy import ndimage
import cv2


def sobel_kernels(img_grayscale: np.ndarray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = lowPassFilters.correlation(img_grayscale, Kx)
    Iy = lowPassFilters.correlation(img_grayscale, Ky)

    magnitude_matrix = np.hypot(Ix, Iy)
    magnitude_matrix = magnitude_matrix / magnitude_matrix.max() * 255
    theta_matrix = np.arctan2(Iy, Ix)

    return (magnitude_matrix, theta_matrix)


def non_max_suppression(gradient_matrix: np.ndarray, theta_matrix: np.ndarray):
    """Check if the pixels on the same direction are more or less intense than the ones being processed. Used with canny detector

    Args:
        gradient_matrix (np.ndarray): magnitude matrix
        theta_matrix (np.ndarray): theta matrix

    Returns:
        _type_: filtered matrix of the same type
    """

    m, n = gradient_matrix.shape
    # Create a matrix initialized to 0 of the same size of the original gradient intensity matrix
    Z = np.zeros((m, n), dtype=np.int32)

    # Identify the edge direction based on the angle value from the angle matrix
    angle = theta_matrix * 180. / np.pi
    angle[angle < 0] += 180

    # Check if the pixel in the same direction has a higher intensity than the pixel that is currently processed
    for i in range(1, m-1):
        for j in range(1, n-1):
            try:
                # Max intensity = 255 for white pixels
                q = 255
                r = 255

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_matrix[i, j+1]
                    r = gradient_matrix[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_matrix[i+1, j-1]
                    r = gradient_matrix[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_matrix[i+1, j]
                    r = gradient_matrix[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_matrix[i-1, j-1]
                    r = gradient_matrix[i+1, j+1]

                # If there are no pixels in the edge direction having more intense values, then the value of the current pixel is kept
                if (gradient_matrix[i, j] >= q) and (gradient_matrix[i, j] >= r):
                    Z[i, j] = gradient_matrix[i, j]
                else:
                    # Intensity value of the current pixel is set to 0
                    Z[i, j] = 0

            except IndexError as e:
                print(f'IndexError: {e}')
                pass

    # Return the image processed with the non-max suppression algorithm
    return Z


def double_threshold(img_grayscale: np.ndarray, low_TH_ratio: float = 0.05, high_TH_ratio: float = 0.09):
    """To identify(filter) the strong, weak and non-relevant pixels 

    Args:
        img (np.ndarray): original image matrix (grayscale)
        low_TH_ratio (float, optional): value of the low threshold. Defaults to 0.05.
        high_TH_ratio (float, optional): value of the high threshold.. Defaults to 0.09.

    Returns:
        _type_: filtered matrix of the same type and shape of the original image
    """

    # Calculating both thresholds
    high_TH = img_grayscale.max() * high_TH_ratio
    low_TH = high_TH * low_TH_ratio

    M, N = img_grayscale.shape
    result_matrix = np.zeros((M, N), dtype=np.int32)

    weak_value = np.int32(25)
    strong_value = np.int32(255)

    # All pixels having intensity higher than high_TH are flagged as strong
    strong_i, strong_j = np.where(img_grayscale >= high_TH)

    # All pixels having intensity between both thresholds are flagged as weak
    weak_i, weak_j = np.where(
        (img_grayscale <= high_TH) & (img_grayscale >= low_TH))

    # All pixels having intensity lower than low_TH are flagged as non-relevant
    zeros_i, zeros_j = np.where(img_grayscale < low_TH)

    result_matrix[strong_i, strong_j] = strong_value
    result_matrix[weak_i, weak_j] = weak_value

    # result_matrix contains only 2 pixel intensity categories (strong and weak)
    return (result_matrix, weak_value, strong_value)


def hysteresis(img_grayscale: np.ndarray, weak_value: int, strong_value: int = 255):
    """Transform weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one

    Args:
        img (np.ndarray): original image (grayscale)
        weak_value (int): low threshold value
        strong_value (int, optional): high threshold value. Defaults to 255.

    Returns:
        np.ndarray: filtered matrix of the same type and shape of the original image
    """

    M, N = img_grayscale.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            # Check for each weak pixel
            if (img_grayscale[i, j] == weak_value):
                try:
                    # Check for the surrounding pixels (box)
                    if ((img_grayscale[i+1, j-1] == strong_value) or (img_grayscale[i+1, j] == strong_value) or (img_grayscale[i+1, j+1] == strong_value)
                        or (img_grayscale[i, j-1] == strong_value) or (img_grayscale[i, j+1] == strong_value)
                            or (img_grayscale[i-1, j-1] == strong_value) or (img_grayscale[i-1, j] == strong_value) or (img_grayscale[i-1, j+1] == strong_value)):
                        img_grayscale[i, j] = strong_value
                    else:
                        # Assign zero to weak pixels
                        img_grayscale[i, j] = 0
                except IndexError as e:
                    print(f'IndexError: {e}')
                    pass
    # return final form of the image using canny edges detector algorithm
    return img_grayscale


def canny_detector(image_path: str):
    """Canny edge detector algorithm

    Args:
        image (np.ndarray): original image (grayscale)

    Returns:
        _type_: matrix of the image with canny mask applied 
    """
    img_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Noise reduction; by applying Gaussian blur to smooth it, where -> kernal 5x5, sigma = 1
    gaussian_mask = lowPassFilters.gaussian_filter(5, 5, 1)
    masked_matrix = lowPassFilters.correlation(img_grayscale, gaussian_mask)

    # Gradient Calculation; by using Sobel kernels
    gradient_matrix, theta_matrix = sobel_kernels(masked_matrix)

    # Thinner edges; by using Non-Maximum Suppression algorithm
    thinner_edges_matrix = non_max_suppression(gradient_matrix, theta_matrix)

    # Double threshold; to identify the pixels of the image according to pre-defined thresholds
    threshold_matrix, weak_value, strong_value = double_threshold(
        thinner_edges_matrix)

    # Hysteresis; to check for surrounding weak pixels if they have a strong value for better edges.
    final_matrix = hysteresis(threshold_matrix, weak_value, strong_value)

    canny_img = final_matrix.astype(np.uint8)

    Helper.store_img_cv2('./output/canny_img.jpg', canny_img)
    return canny_img


def sobel_detector(image_path: str):
    """Detect edges using Sobel algorithm:

    Args:
        img_grayscale (np.ndarray): img converted to grayscale 

    Returns:
        _type_: matrix of the image with sobel mask applied 
    """
    img_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    magnitude_matrix, _ = sobel_kernels(img_grayscale)
    sobel_img = magnitude_matrix.astype(np.uint8)

    Helper.store_img_cv2('./output/sobel_img.jpg', sobel_img)
    return sobel_img


def roberts_detector(image_path: str):
    """Roberts edge detector algorithm

    Args:
        img_grayscale (np.ndarray):original image (grayscale)

    Returns:
        _type_: matrix of the image with Roberts mask applied 
    """
    img_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize pair of  Roberts cross operator
    roberts_cross_v = np.array([[1, 0], [0, -1]])
    roberts_cross_h = np.array([[0, 1], [-1, 0]])

    img_grayscale = img_grayscale.astype('float64')
    img_grayscale /= 255.0

    # Calculate gradient components (Gx, Gy)
    Gy = ndimage.convolve(img_grayscale, roberts_cross_v)
    Gx = ndimage.convolve(img_grayscale, roberts_cross_h)

    # Calculate G (Gradient Magnitude)
    edged_matrix = np.sqrt(np.square(Gx) + np.square(Gy))
    edged_matrix *= 255

    roberts_img = edged_matrix.astype(np.uint8)

    Helper.store_img_cv2('./output/roberts_img.jpg', roberts_img)
    return roberts_img


def prewitt_detector(image_path: str):
    """Prewitt edge detector algorithm

    Args:
        img_grayscale (np.ndarray):original image (grayscale)

    Returns:
        _type_: matrix of the image with Prewitt mask applied 
    """

    img_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

    Gx = lowPassFilters.correlation(img_grayscale, Kx)
    Gy = lowPassFilters.correlation(img_grayscale, Ky)

    # Calculate G (Gradient Magnitude)
    magnitude_matrix = np.hypot(Gx, Gy)
    magnitude_matrix = magnitude_matrix / magnitude_matrix.max() * 255

    prewitt_img = magnitude_matrix.astype(np.uint8)

    Helper.store_img_cv2('./output/prewitt_img.jpg', prewitt_img)
    return prewitt_img
