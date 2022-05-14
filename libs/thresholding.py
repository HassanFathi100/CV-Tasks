from PIL import ImageFilter
import numpy as np
# from normalization import normalize_histogram
from helper import store_img_pil


def otsu_calculations(image_grayscale_PIL):

    image = image_grayscale_PIL

    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    blurred_image = np.copy(blurred_image)
    # store_img_pil(blurred_image, './output/blurred_image_pil.jpg')

    _, _, count_intensity = normalize_histogram(
        blurred_image)

    cumulative_intensity = count_intensity.cumsum()
    cumulative_intensity = cumulative_intensity / max(cumulative_intensity)
    cumulative_intensity = np.round_(cumulative_intensity, decimals=4)

    new_count_intensity = count_intensity / max(count_intensity)

    bins = np.arange(256)
    fn_min = np.inf
    threshold = -1

    for i in range(1, 256):
        p1, p2 = np.hsplit(new_count_intensity, [i])  # probabilities
        # cum sum of classes
        q1, q2 = cumulative_intensity[i], cumulative_intensity[255] - \
            cumulative_intensity[i]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            threshold = i
    # find otsu's threshold value with OpenCV function
    return threshold


def local_thresholding(resized_gray_image, window_factor: int = 16):
    # resized_gray_image of type:  <class 'PIL.Image.Image'>
    input_image = resized_gray_image

    threshold = otsu_calculations(input_image)
    # print(f'Threshold returned from otsu calculations: {threshold}')

    if threshold > 150:
        threshold = 20
        # print(f'modified threshold: {threshold}')
    input_image = np.array(input_image)
    height, width = input_image.shape

    sample_window_1 = width / window_factor
    sample_window_2 = sample_window_1 / 2

    integrated_image = np.zeros_like(input_image, dtype=np.uint32)

    # integrate_image
    for column in range(width):
        for row in range(height):
            integrated_image[row,
                             column] = input_image[0:row, 0:column].sum()

    output_image = np.zeros_like(input_image)

    # Adaptive threshold algorithm
    for column in range(width):
        for row in range(height):
            y1 = round(max(row-sample_window_2, 0))
            y2 = round(min(row+sample_window_2, height-1))
            x1 = round(max(column-sample_window_2, 0))
            x2 = round(min(column+sample_window_2, width-1))

            count = (y2-y1)*(x2-x1)

            total = integrated_image[y2, x2]-integrated_image[y1,
                                                              x2]-integrated_image[y2, x1]+integrated_image[y1, x1]

            if np.all(input_image[row, column]*count < total*(100-threshold)/100):
                output_image[row, column] = 0
            else:
                output_image[row, column] = 255
    store_img_pil('./output/local_thresholding.bmp', output_image)
    return output_image


def global_thresholding(img_grayscale: np.ndarray, threshold: int):
    image = np.copy(img_grayscale)
    image[image > threshold] = 255
    image[image != 255] = 0
    store_img_pil('./output/global_thresholding.bmp', image)
    return image
