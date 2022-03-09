"""Functions that may help generally.
    PS:
        Function names should be lowercase, with words separated by underscores as necessary to improve readability.

        Variable names follow the same convention as function names.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def store_img(filepath: str, img: np.ndarray):
    cv2.imwrite(filepath, img)
    filename = filepath.split('output/')[1]
    print(f'{filename} saved successfully in output directory.')


def resize_img(img: np.ndarray, x_scale: float = 0.5, y_scale: float = 0.5):
    img = cv2.resize(img, (0, 0), fx=x_scale, fy=y_scale)
    return img


def plot_img(img: np.ndarray):

    final_img = img.astype(np.uint8)
    cv2.imshow(f'img', final_img)
    cv2.waitKey(0)


def plot_histogram(intensity_values: list, intensity_counter: int):
    """plot histogram

    Args:
        intensity_values (np.array): intensity values
        intensity_counter (int): number of pixels for each intensity level
    """

    plt.stem(intensity_values, intensity_counter)
    plt.xlabel('intensity value')
    plt.ylabel('number of pixels')
    plt.title('Histogram')
    plt.show()
    plt.savefig('./output/histogram.png', bbox_inches='tight')
    print('histogram.png saved successfully in output directory.')


def plot_normalized(intensity_values: list, intensity_counter: int):
    """plot normalized curve

    Args:
        intensity_values (np.array): intensity values
        intensity_counter (int): number of pixels for each intensity level
    """

    plt.plot(intensity_values, intensity_counter)
    plt.xlabel('intensity value')
    plt.ylabel('number of pixels')
    plt.title('Normalized Histogram')
    plt.show()
    plt.savefig('./output/normalized_histogram.png', bbox_inches='tight')
    print('normalized_histogram.png saved successfully in output directory.')
