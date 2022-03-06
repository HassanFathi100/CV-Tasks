import matplotlib.pyplot as plt

# function to obtain histogram of an image
def calculate_histo(img, m, n):
    """
        m: number of rows
        n: number of columns

        get m, n from img.shape method.
    """ 
    # intensity values
    intensity_levels = []

    # number of pixels for each intensity
    count_intensity =[]
      
      
    # calculate intensity values
    for intensity in range(0, 256):
        intensity_levels.append(intensity)
        temp = 0
          
        # loops on each pixel
        for i in range(m):
            for j in range(n):
                if img[i, j]== intensity:
                    temp+= 1
        count_intensity.append(temp)
          
    return (intensity_levels, count_intensity)


def plot_histogram(r, count):
    
    """
        r , count are the output of calculating histogram function
        r: intensity values
        count: the number of pixels for each intensity level
    """ 
    
    plt.stem(r, count)
    plt.xlabel('intensity value')
    plt.ylabel('number of pixels')
    plt.title('Histogram')
