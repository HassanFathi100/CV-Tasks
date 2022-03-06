"""Histogram equalization"""

import cv2
import numpy as np
import histogram as hist
from matplotlib import pyplot as plt


img = cv2.imread('./assets/apple.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0 ,0), fx = 0.5, fy= 0.5)

rows, columns = img.shape

intensityValues, intensityCount = hist.calculate_histogram(img, rows, columns)
hist.plot_histogram(intensityValues, intensityCount)

# print(intensityValues)
# print(intensityCount)


# plt.plot(intensityValues, intensityCount)
# plt.show()


# Already defined equalization function
# equalizedImg = cv2.equalizeHist(img)

cv2.imshow('original', img)
cv2.imwrite('filename', img)
cv2.waitKey(0)
cv2.destroyAllWindows()