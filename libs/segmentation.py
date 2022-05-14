import cv2
import matplotlib.pyplot as plt
import numpy as np
from thresholding import global_thresholding



def apply_optimal_threshold(source: np.ndarray):
    """
    Applies Thresholding To The Given Grayscale Image Using The Optimal Thresholding Method
    :param source: NumPy Array of The Source Grayscale Image
    :return: Thresholded Image
    """

    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Calculate Initial Thresholds Used in Iteration
    print(f"src in optimal: {src}")
    print(f"src shape: {src.shape}")
    OldThreshold = GetInitialThreshold(src)
    NewThreshold = GetOptimalThreshold(src, OldThreshold)
    iteration = 0
    # Iterate Till The Threshold Value is Constant Across Two Iterations
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = GetOptimalThreshold(src, OldThreshold)
        iteration += 1
    # src[src >= 25] = 0
    # Return Thresholded Image Using Global Thresholding
    return global_thresholding(src, NewThreshold)


def GetInitialThreshold(source: np.ndarray):
    """
    Gets The Initial Threshold Used in The Optimal Threshold Method
    :param source: NumPy Array of The Source Grayscale Image
    :return Threshold: Initial Threshold Value
    """
    # Maximum X & Y Values For The Image
    MaxX = source.shape[1] - 1
    MaxY = source.shape[0] - 1
    # Mean Value of Background Intensity, Calculated From The Four Corner Pixels
    BackMean = (int(source[0, 0]) + int(source[0, MaxX]) + int(source[MaxY, 0]) + int(source[MaxY, MaxX])) / 4
    Sum = 0
    Length = 0
    # Loop To Calculate Mean Value of Foreground Intensity
    for i in range(0, source.shape[1]):
        for j in range(0, source.shape[0]):
            # Skip The Four Corner Pixels
            if not ((i == 0 and j == 0) or (i == MaxX and j == 0) or (i == 0 and j == MaxY) or (
                    i == MaxX and j == MaxY)):
                Sum += source[j, i]
                Length += 1
    ForeMean = Sum / Length
    # Get The Threshold, The Average of The Mean Background & Foreground Intensities
    Threshold = (BackMean + ForeMean) / 2
    return Threshold


def GetOptimalThreshold(source: np.ndarray, Threshold):
    """
    Calculates Optimal Threshold Based on Given Initial Threshold
    :param source: NumPy Array of The Source Grayscale Image
    :param Threshold: Initial Threshold
    :return OptimalThreshold: Optimal Threshold Based on Given Initial Threshold
    """
    # Get Background Array, Consisting of All Pixels With Intensity Lower Than The Given Threshold
    Back = source[np.where(source < Threshold)]
    # Get Foreground Array, Consisting of All Pixels With Intensity Higher Than The Given Threshold
    Fore = source[np.where(source > Threshold)]
    # Mean of Background & Foreground Intensities
    BackMean = np.mean(Back)
    ForeMean = np.mean(Fore)
    # Calculate Optimal Threshold
    OptimalThreshold = (BackMean + ForeMean) / 2
    return OptimalThreshold


def LocalThresholding(source: np.ndarray, RegionsX: int, RegionsY: int, ThresholdingFunction):
    """
       Applies Local Thresholding To The Given Grayscale Image Using The Given Thresholding Callback Function
       :param source: NumPy Array of The Source Grayscale Image
       :param Regions: Number of Regions To Divide The Image To
       :param ThresholdingFunction: Function That Does The Thresholding
       :return: Thresholded Image
       """
    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    YMax, XMax = src.shape
    Result = np.zeros((YMax, XMax))
    YStep = YMax // RegionsY
    XStep = XMax // RegionsX
    XRange = []
    YRange = []
    for i in range(0, RegionsX):
        XRange.append(XStep * i)

    for i in range(0, RegionsY):
        YRange.append(YStep * i)

    XRange.append(XMax)
    YRange.append(YMax)
    for x in range(0, RegionsX):
        for y in range(0, RegionsY):
            Result[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = ThresholdingFunction(src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]])
    return Result






def ApplyThreshold():
   
    source = cv2.imread("./assets/assetsNew/hand_512.jpg")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    img = np.copy(source)
    OptImg = apply_optimal_threshold(img)
    cv2.imshow('Optimal thresholded img', OptImg)
    cv2.waitKey()

if __name__ == "__main__":
    ApplyThreshold()