"""Functions that may help generally.
    PS:
        Function names should be lowercase, with words separated by underscores as necessary to improve readability.

        Variable names follow the same convention as function names.
"""
import cv2

# function to store an img to local hard


def store_img(name, img):
    cv2.imwrite(name, img)

#  read function
# plot function
#  resize window function
