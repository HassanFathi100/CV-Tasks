import helper as Helper
import numpy as np
import math
import frequency_domain_filters as filter
import cv2


def square_image(img: np.ndarray):
    """_make the image square (height = width)_

    Args:
        img (np.ndarray): _the image to square_
        
    Return:
        the image with black square fit
    """
    
    height, width  = img.shape
    
    if(height > width):
        paddingValue = math.floor((height - width)/2)
        paddedImage = np.pad(img, [(paddingValue, paddingValue), (0, 0)], 'constant', constant_values=(0))
    elif(width > height):
        paddingValue = math.floor((width - height)/2)
        paddedImage = np.pad(img, [(paddingValue, paddingValue), (0, 0)], 'constant', constant_values=(0))
    else: 
        paddedImage = img
    
    return paddedImage


def hybrid_image(img1: np.ndarray, img2: np.ndarray):
    """
    _Calculate the hybrid image_

    Args:
        img1 (np.ndarray): _high frequency image_
        img2 (np.ndarray): _low frequency image_
        
    Return:
        hybrid image array
    """
    
    highFrequencyImg = filter.filter_fun(img1, 25, "hpf")
    lowFrequencyImg = filter.filter_fun(img2, 25, "lpf")
    
    hybridImg = highFrequencyImg + lowFrequencyImg
    
    return hybridImg
    
    


#==============================================
#Pass the 2 photos from main like this
dog = cv2.imread("./assets/dog.jpg", 1)
cat = cv2.imread("./assets/cat.jpg", 1)


hybrid = hybrid_image(dog, cat)
Helper.store_img_cv2("./output/hybrid-dog-cat.jpg", hybrid.astype(np.uint8))
#==============================================