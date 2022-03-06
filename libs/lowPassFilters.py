"""Implement here functions that filter the noisy images using the following low pass filters:
    Average, Gaussian and median filters.
    """
import cv2
import numpy as np
import numpy
from PIL import Image
import cv2
def AverageFilter ():
    img = cv2.imread('./assets/apple.jpg', cv2.IMREAD_GRAYSCALE)
    
    m, n = img.shape
    
    # Develop Averaging filter(3, 3) mask
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9
    
    # Convolving the 3X3 mask over the input image
    img_new = np.zeros([m, n])
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
            
            img_new[i, j]= temp
            
    img_new = img_new.astype(np.uint8)
    cv2.imshow("dst ", img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def MedianFilter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
            
    return data_final


def main():
    img = Image.open('./assets/apple.jpg').convert("L")
    arr = numpy.array(img)
    removed_noise = MedianFilter(arr, 6) 
    img = Image.fromarray(removed_noise)
    img.show()


    

# main()
# AverageFilter()