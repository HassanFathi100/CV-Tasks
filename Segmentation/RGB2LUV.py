from typing import Tuple
import numpy as np
import cv2

np.seterr(invalid='ignore')

def RGB2XYZ(image):
    """
    Converts an RGB image to XYZ.

    Args:
        image: An RGB image.

    Returns:
        A tuple containing an X image, a Y image, and a Z image.
    """
    
    image = image / 255.0
    
    X = np.dot(image, [0.412453, 0.357580, 0.180423])
    Y = np.dot(image, [0.212671, 0.715160, 0.072169])
    Z = np.dot(image, [0.019334, 0.119193, 0.950227])

    return X, Y, Z

def XYZ2LUV(X, Y, Z):
    """
    Converts an XYZ image to LUV.

    Args:
        X: An X image.
        Y: A Y image.
        Z: A Z image.

    Returns:
        An LUV image.
    """

    # These are Calculated using the same equations as u_dash and v_dash
    # with X, Y, Z being the reference values got from a lookup table
    # https://www.easyrgb.com/en/math.php
    # for observer Illumination being D65 and a CIE 2° Illuminant.
    U_REF = 0.19793943
    V_REF = 0.46831096

    L = np.where((Y / 100) > 0.008856, 116 * np.power((Y / 9), 1 / 3) - 16, 903.3 * (Y / 9))

    u_dash = np.where((X + (15 * Y) + (3 * Z)) != 0, np.divide((4.0 * X), (X + (15.0 * Y) + (3.0 * Z))), 0)
    v_dash = np.where((X + (15 * Y) + (3 * Z)) != 0, np.divide((9.0 * Y), (X + (15.0 * Y) + (3.0 * Z))), 0)

    U = 13 * L * (u_dash - U_REF)
    V = 13 * L * (v_dash - V_REF)

    L = (255.0 / 100) * L
    U = (255.0 / 354) * (U + 134)
    V = (255.0 / 262) * (V + 140)

    LUV = np.dstack((L, U, V)).astype(np.uint8)

    return LUV

def RGB2LUV(image):
    """
    Converts an RGB image to LUV.

    Args:
        image: An RGB image.

    Returns:
        A LUV image.
    """
    X, Y, Z = RGB2XYZ(image)
    return XYZ2LUV(X, Y, Z)

def ShowLUV(image):
    cv2_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    LUV = RGB2LUV(rgb)

    # For some reason the LUV values are not the same as the LUV values got from online websites for color space conversion.
    # https://colormine.org/convert/rgb-to-luv
    # and after mapping it to be from 0 to 255 space, it does not match that of the CV2 library.
    # mapping is done following the formula in the link bellow.
    # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#MathJax-Element-56-Frame
    print(image[:1, :1, :])
    print(cv2_luv[:1, :1, :])
    print(LUV[:1, :1, :])

    cv2.imshow('LUV', LUV)
    cv2.imshow('CV LUV', cv2_luv)
    cv2.waitKey(0)