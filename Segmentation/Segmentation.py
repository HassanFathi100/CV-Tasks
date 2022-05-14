from helpers import Point, regionGrow
from classes import AgglomerativeClustering, KMeans, MeanShift
import numpy as np
import cv2
np.random.seed(42)

def apply_k_means(source, k=5, max_iter=100):
    """Segment image using K-means

    Args:
        source (nd.array): BGR image to be segmented
        k (int, optional): Number of clusters. Defaults to 5.
        max_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        segmented_image (nd.array): image segmented
        labels (nd.array): labels of every point in image
    """
    # convert to RGB
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # reshape image to points
    pixel_values = source.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # run clusters_num-means algorithm
    model = KMeans(K=k, max_iters=max_iter)
    y_pred = model.predict(pixel_values)

    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)

    # flatten labels and get segmented image
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(source.shape)

    return segmented_image, labels


def apply_region_growing(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)
    img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    seeds = []
    for i in range(3):
        x = np.random.randint(0, img_gray.shape[0])
        y = np.random.randint(0, img_gray.shape[1])
        seeds.append(Point(x, y))

    # seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
    output_image = regionGrow(img_gray, seeds, 10)

    return output_image


def apply_agglomerative(source: np.ndarray, clusters_numbers: int = 2, initial_clusters: int = 25):
    """

    :param source:
    :param clusters_numbers:
    :param initial_clusters:
    :return:
    """
    agglomerative = AgglomerativeClustering(source=source, clusters_numbers=clusters_numbers,
                                            initial_k=initial_clusters)

    return agglomerative.output_image


def apply_mean_shift(source: np.ndarray, threshold: int = 60):
    """

    :param source:
    :param threshold:
    :return:
    """

    src = np.copy(source)

    ms = MeanShift(source=src, threshold=threshold)
    ms.run_mean_shift()
    output = ms.get_output()

    return output