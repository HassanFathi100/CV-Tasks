from PIL import Image, ImageOps
import numpy as np

image_path = './assets/apple.jpg'

# creating an og_image object
og_image = Image.open(image_path)
gray_image = ImageOps.grayscale(og_image)
# Convert it to numpy array
image = np.array(gray_image)


def global_thresholding(img_grayscale: np.ndarray, threshold: int):
    image = np.copy(img_grayscale)
    image[image > threshold] = 255
    image[image != 255] = 0
    return image


def local_thresholding(img_grayscale: np.ndarray):

    regions = 2
    v_splits = np.array_split(img_grayscale, regions)
    splits = []
    output = None

    for chunk in v_splits:
        splits.append(np.split(chunk, regions, -1))

    c1 = []
    # Calculate the mean and threshold for each split
    for ix, x in enumerate(splits):
        for iy, y in enumerate(x):
            threshold = int(np.mean(y))
            splits[ix][iy] = global_thresholding(splits[ix][iy], threshold)
        c1.append(np.concatenate(splits[ix], -1))

    output = np.concatenate(c1)

    return output
