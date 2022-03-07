"""Histogram normalization
    """
import histogram as hist
import numpy as np
import matplotlib.pyplot as plt

def normalize_histogram(img):
    img_array = np.array(img)

    #get minimum and maximum pixel value in the image
    minimum_value = np.min(img_array)
    maximum_value = np.max(img_array)

    #normalize equation
    normalized_img = (img_array - minimum_value) * (1.0 / (maximum_value - minimum_value))
    m, n = normalized_img.shape

    intensity_levels, count_intensity = hist.calculate_histogram(normalized_img, m, n)

    return normalized_img, intensity_levels, count_intensity

def plot_normalized(r, count):
    
    """
        r , count are the output of calculating histogram function
        r: intensity values
        count: the number of pixels for each intensity level
    """ 
    plt.plot(r, count)
    plt.xlabel('intensity value')
    plt.ylabel('number of pixels')
    plt.title('Normalized Histogram')
    plt.show()