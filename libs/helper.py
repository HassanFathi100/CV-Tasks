"""Functions that may help generally.
    PS:
        Function names should be lowercase, with words separated by underscores as necessary to improve readability.

        Variable names follow the same convention as function names.
"""

import numpy as np
import matplotlib.pyplot as plt


from PIL import Image


def store_img_pil(filepath: str, img: np.ndarray):
    result = Image.fromarray((img).astype(np.uint8))
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


def resize_img(img: np.ndarray, basewidth: int = 300):
    w_percent = (basewidth/float(img.size[0]))
    h_size = int((float(img.size[1])*float(w_percent)))
    resized_img = img.resize((basewidth, h_size), Image.ANTIALIAS)
    return resized_img


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


def plotRGBvsGray(rgb_image, gray_scale_image):
    # PLOT THE IMAGES
    fig = plt.figure(1)
    og_image, gray_image = fig.add_subplot(121), fig.add_subplot(122)
    og_image.imshow(rgb_image)
    gray_image.imshow(gray_scale_image, cmap=plt.cm.get_cmap('gray'))
    fig.show()
    plt.savefig('./output/RGBvsGray.png', bbox_inches='tight')
    plt.show()


# def show_images(images: list[np.ndarray], title: str, labels: list[str]) -> None:
#     n: int = len(images)
#     f = plt.figure()

#     f.suptitle(title)
#     for i in range(n):
#         # Debug, plot figure
#         f.add_subplot(1, n, i + 1)
#         # f.set_label(labels[i])
#         plt.imshow(images[i], cmap=plt.cm.get_cmap('gray'))
#         plt.axis('off')
#         plt.title(labels[i])
#     plt.show(block=True)
