"""Functions that may help generally.
    PS:
        Function names should be lowercase, with words separated by underscores as necessary to improve readability.

        Variable names follow the same convention as function names.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image, ImageOps


def store_img_cv2(filepath: str, img: np.ndarray):
    cv2.imwrite(filepath, img)
    filename = filepath.split('output/')[1]
    print(f'{filename} saved successfully in output directory.')


def store_img_pil(img_matrix, filepath):
    result = Image.fromarray((img_matrix).astype(np.uint8))
    result.save(filepath)
    filename = filepath.split('output/')[1]
    print(f'{filename} saved successfully in output directory.')


def store_img_plt(filepath: str, img: np.ndarray):
    filename = filepath.split('output/')[1]
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight')
    print(f'{filename} saved successfully in output directory.')


def resize_img(img: np.ndarray, x_scale: float = 0.5, y_scale: float = 0.5):
    img = cv2.resize(img, (0, 0), fx=x_scale, fy=y_scale)
    return img


def showing(img):
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    img = np.array(img, dtype=float)/float(255)
    cv2.imshow('test', img)
    cv2.resizeWindow('test', 600, 600)
    cv2.waitKey(0)


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
    plt.savefig('../output/histogram.png', bbox_inches='tight')
    plt.show()
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
    plt.savefig('./output/normalized_histogram.png', bbox_inches='tight')
    plt.show()
    print('normalized_histogram.png saved successfully in output directory.')
