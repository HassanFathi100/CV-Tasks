from libs import histogram, edgeDetection, equalization, noise, normalization, lowPassFilters

image_path = './assets/lion.jpg'

# Calling any function saves output images in the output directory


def additive_noise():
    noise.gaussian_noise(image_path)
    noise.uniform_noise(image_path)
    noise.s_and_p_noise(image_path)


def low_pass_filters():
    lowPassFilters.average_filter(image_path)
    lowPassFilters.apply_gaussian_filter(image_path)
    # lowPassFilters.median_filter(imgpath)


def edge_detection():
    # edgeDetection.sobel_detector(image_path)
    # edgeDetection.prewitt_detector(image_path)
    # edgeDetection.roberts_detector(image_path)
    edgeDetection.canny_detector(image_path)


def draw_curvers():
    histogram.calculate_histogram(image_path)


def equalize_img():
    equalization.equalization(image_path)


def normalize_img():
    normalization.normalize_histogram(image_path)


# Run script
# additive_noise()
# low_pass_filters()
edge_detection()
# draw_curvers()
# equalize_img()
# normalize_img()
