import cv2
import matplotlib.pyplot as plt
import numpy as np
import thresholding  
from PIL import ImageFilter
from edgeDetection import double_threshold



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
    return thresholding.global_thresholding(src, NewThreshold)


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

def apply_otsu_threshold(source :np.ndarray):
    """
     Applies otsue global Thresholding for greyscale image
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    input_image = np.copy(source)

    if len(input_image.shape) > 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        pass
    
    threshold = thresholding.otsu_calculations(input_image)
    return thresholding.global_thresholding(input_image,threshold)


def apply_spectral_threshold(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Spectral Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Get Image Dimensions
    YRange, XRange = src.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(src.ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (YRange * XRange)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            try:
                # Background Intensities Array
                Back = np.arange(0, LowT)
                # Low Intensities Array
                Low = np.arange(LowT, HighT)
                # High Intensities Array
                High = np.arange(HighT, 256)
                # Get Low Intensities CDF
                CDFL = np.sum(PDF[LowT:HighT])
                # Get Low Intensities CDF
                CDFH = np.sum(PDF[HighT:256])
                # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
                BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                HighMean = sum(High * PDF[HighT:256]) / CDFH
                # Calculate Cross-Class Variance
                Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                        CDFH * (HighMean - GMean) ** 2))
                # Filter Out Max Variance & It's Threshold
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowT
                    OptimalHigh = HighT
            except RuntimeWarning:
                pass
    return double_threshold(src, OptimalLow, OptimalHigh, 128, False)


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
   
    img = cv2.imread("./assets/assetsNew/hand_512.png",1)
    
    OptImg = apply_optimal_threshold(img)
    OtsuImg = apply_otsu_threshold(img)
    spec_img = apply_spectral_threshold(img)
    cv2.imshow('Optimal thresholded img', OptImg)
    cv2.imshow('Otsuimal thresholded img', OtsuImg)
    cv2.imshow('spectimal thresholded img', spec_img)
    cv2.waitKey(5000)

if __name__ == "__main__":
    ApplyThreshold()