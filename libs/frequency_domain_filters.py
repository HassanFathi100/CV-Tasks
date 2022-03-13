import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import helper as Helper

# img = cv2.imread('./assets/lion.jpg',)
# img = cv2.resize(img, (0,0), fx= 0.5, fy= 0.5)

def frequency_domain_fun(image):
    """
    padding function to make size of mask the same of image to keep the size 
    of image the same as before applying filtering
    """ 
    src=np.copy(image)
    # if(len(image.shape)<3):
    #   print ('gray')
    #   grayscale_image=image
    # elif len(image.shape)==3:
    #   print ('Color(RGB)')
    grayscale_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    #getting fft image
    fft_image= np.fft.fft2(grayscale_image)
    shfft_image=np.fft.fftshift(fft_image)
    return shfft_image

  

def padding_fun(image ,width , height,padding_value):
    img_width,img_height=image.shape
    right_width_added=(width-img_width)//2
    left_width_added=width-(right_width_added+img_width)
    upper_height_added=(height-img_height)//2
    lower_height_added=height-(img_height+upper_height_added)
    padding_img= np.pad(image,((right_width_added,left_width_added),(upper_height_added,lower_height_added)),
    constant_values=padding_value)
    print(width,height)
    print(padding_img.shape)
    return padding_img


"""
filter_fun is a function take arguments : original image , mask size which applied in frequency_domain_image
and filter_type as a string to select low or high pass filter 
in low pass filter case :making zero mask of specific size then getting pad with constant value "1"
with the same size of original image
in high pass filter case :making ones mask of specific size then getting pad with constant value "0"
with the same size of original image

"""
def filter_fun(image ,size,filter_type):
    row,col,ch=image.shape
    frequency_domain_image=frequency_domain_fun(image)
    if filter_type=="lpf":
        mask=np.ones((size,size))
        padding_mask=padding_fun(mask,row,col,0)
    elif filter_type=="hpf":
        mask=1-np.ones((size,size))
        padding_mask=padding_fun(mask,row,col,1)
        
    filtered_image=np.fft.ifftshift(frequency_domain_image*padding_mask)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image).clip(0,255).astype(np.uint8)

    # cv2.imwrite('../gaussian_filtered_img.jpg', filtered_image)
    
    return filtered_image
# img2=filter_fun(img ,100,"low_pass_filter")
# print(img2)

# Helper.showing(img2)
# cv2.waitKey(0)



