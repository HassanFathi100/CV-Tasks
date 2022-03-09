from libs import edgeDetection, equalization, noise, normalization, lowPassFilters


from PIL import Image, ImageOps
import numpy as np

image_path = './assets/lion.jpg'

# creating an og_image object
og_image = Image.open(image_path)
gray_image = ImageOps.grayscale(og_image)
# Convert it to numpy array
image = np.array(gray_image)


def additive_noise():
    noise.gaussian_noise(image_path)
    noise.uniform_noise(image_path)
    noise.s_and_p_noise(image_path)


def low_pass_filters():
    lowPassFilters.average_filter(image_path)
    lowPassFilters.apply_gaussian_filter(image_path)
    # lowPassFilters.median_filter(imgpath)


def edge_detection():
    edgeDetection.sobel_detector(image_path)
    edgeDetection.prewitt_detector(image_path)
    edgeDetection.roberts_detector(image_path)
    edgeDetection.canny_detector(image_path)


def equalize_img():
    equalization.equalization(image_path)


def normalize_img():
    normalization.normalize_histogram(image_path)


# Run script

additive_noise()
low_pass_filters()
edge_detection()
equalize_img()
normalize_img()
