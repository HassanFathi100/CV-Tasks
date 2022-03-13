from matplotlib.image import imread
import matplotlib.pyplot as plt
from .histogram import calculate_histogram
from .helper import plotRGBvsGray

def rgb2Grey(image_path: str):
    """Converts RGB image into a gray scale image

    Args:
        image_path (str): the input image 

    Returns:
        gray_scale_image (np.ndarray): converted image (grayscale)
    """
    
    # Read the RGB image
    rgb_image = imread(image_path)

    ## Separate the RGB channels
    r,g,b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]

    ## Convertion equation constants:
    gamma = 1.04 # controls the brightness of the image
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722 # channels constants

    gray_scale_image = (r_const*r)**gamma + (g_const*g)**gamma + (b_const*b)**gamma

    plotRGBvsGray(rgb_image, gray_scale_image)
    plot_RGB_Histo(rgb_image)

    return gray_scale_image


def plot_RGB_Histo(rgb_image):
    RGB = [rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]]
    color = ('b','g','r')

    plt.figure()
    for i,col in enumerate(color):
        histr = calculate_histogram(RGB[i])
        plt.plot(histr, color = col)
        # plt.xlim([0,1])
    plt.savefig('./output/RGB_Histo.png', bbox_inches='tight')
    plt.show()