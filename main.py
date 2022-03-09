"""
Main sequence:
    - Read the image using cv2 and conver it to grayscale
    - Add noise to the image
    - Filter the noisy image (weird sequence)
    - Detect edges in the image
    - Draw histogram and distribution curves
    - Equalize and normalize the image
    """

import cv2
from libs import histogram, edgeDetection, equalization, noise, normalization, lowPassFilters

imgpath = './assets/lion.jpg'

# Read image using Cv2
img_original = cv2.imread(imgpath)

# Convert image to grayscale
img_grayscale = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

# Get rows and columns numbers of the image
rows, columns = img_grayscale.shape


# Calling any function saves output images in the output directory

def additive_noise():
    noise.gaussian_noise(img_original)
    noise.uniform_noise(img_original)
    noise.s_and_p_noise(img_grayscale)


def low_pass_filters():
    lowPassFilters.average_filter(img_grayscale)
    lowPassFilters.apply_gaussian_filter(img_grayscale)
    # lowPassFilters.median_filter(imgpath)


def edge_detection():
    edgeDetection.sobel_detector(img_grayscale)
    edgeDetection.prewitt_detector(img_grayscale)
    edgeDetection.roberts_detector(img_grayscale)
    edgeDetection.canny_detector(img_grayscale)


def draw_curvers():
    histogram.calculate_histogram(img_grayscale, rows, columns)


def equalize_img():
    equalization.equalization(img_grayscale)


def normalize_img():
    normalization.normalize_histogram(img_grayscale)


# Run script
additive_noise()
low_pass_filters()
edge_detection()
draw_curvers()
equalize_img()
normalize_img()
