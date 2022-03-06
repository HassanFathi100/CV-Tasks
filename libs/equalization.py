"""Histogram equalization"""

import cv2
import numpy as np
import histogram as hist
from matplotlib import pyplot as plt

# Reading image
img = cv2.imread('./assets/apple.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0 ,0), fx = 0.5, fy= 0.5)

rows, columns = img.shape

# Histogram
intensityValues, intensityCount = hist.calculate_histogram(img, rows, columns)

# Converting list to numpay array
intensityCountArray = np.array(intensityCount)

# Get the sum of all bins
intensityCountSum = np.sum(intensityCountArray)

# Calculating propability density function
PDF = intensityCountArray/intensityCountSum

# Calculating Cumulative density function
CDF = np.array([])
CDF = np.cumsum(PDF)

# Rounding CDF values
equalizedHistogram = np.round((255 * CDF), decimals = 0)

# Flattening the image 
imgVector = img.ravel()

# Converting 1D array (vector) to 2D array (image)
mappedImgVector = []
for pixel in imgVector:
    mappedImgVector.append(equalizedHistogram[pixel])

finalImg = np.reshape(np.asarray(mappedImgVector), img.shape).astype(np.uint8)


# Already defined equalization function
# equalizedImg_alreadyDefine = cv2.equalizeHist(img)

cv2.imshow('Equalized Image', finalImg)
cv2.waitKey(0)
cv2.destroyAllWindows()