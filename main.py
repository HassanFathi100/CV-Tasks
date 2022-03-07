"""
Main sequence:
    - Read the image
    - Convert the image to grayscale
    - Store the image as an array (pixels)
    - Add noise to the image
    - Filter the noisy image (weird sequence)
    - Detect edges in the image
    - Draw histogram and distribution curves
    - Equalize and normalize the image
    """

from libs import curves, edgeDetection, equalization, helper, lowPassFilters, noise, normalization


def HelloTeam():
    print("Hello Team!")
    
lowPassFilters.average_filter()