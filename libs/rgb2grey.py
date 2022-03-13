from PIL import Image
import matplotlib.pyplot as plt

def rgb2Grey(image_path: str):
    """Converts RGB image into a gray scale image

    Args:
        image_path (str): the input image 

    Returns:
        gray_scale_image (np.ndarray): converted image (grayscale)
    """
    
    # Read the RGB image
    rgb_image = Image.open(image_path)

    ## Separate the RGB channels
    r,g,b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]

    ## Convertion equation constants:
    gamma = 1.04 # controls the brightness of the image
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722 # channels constants

    gray_scale_image = (r_const*r)**gamma + (g_const*g)**gamma + (b_const*b)**gamma

    #####################################################
    # PLOT THE IMAGES
    fig = plt.figure(1)
    og_image, gray_image = fig.add_subplot(121), fig.add_subplot(122)
    og_image.imshow(rgb_image)
    gray_image.imshow(gray_scale_image, cmp = plt.cm.get_cmap('gray'))
    fig.show()
    plt.show()
    ######################################################

    return gray_scale_image