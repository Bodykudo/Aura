import numpy as np
import cv2
from numba import jit, prange


@jit(nopython=True)
def calculate_distance(a, b):
    return np.sqrt(
        np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2) + np.power(a[2] - b[2], 2)
    )


@jit(nopython=True)
def merge_clusters(a, b):
    r = (a[0] + b[0]) // 2
    g = (a[1] + b[1]) // 2
    b = (a[2] + b[2]) // 2
    return (r, g, b)


@jit(nopython=True, parallel=True)
def find_closest_clusters(clusters):
    min_dist = 1e10  # A large float value
    min_i, min_j = 0, 0
    for i in prange(len(clusters)):
        for j in prange(i + 1, len(clusters)):
            dist = calculate_distance(clusters[i], clusters[j])
            if dist < min_dist:
                min_dist = dist
                min_i, min_j = i, j
    return min_i, min_j


@jit(nopython=True)
def find_nearest_cluster(pixel, clusters):
    min_cluster = 0
    min_dist = 1e10  # A large float value
    for k in prange(len(clusters)):
        dist = calculate_distance(pixel, clusters[k])
        if dist < min_dist:
            min_dist = dist
            min_cluster = k
    return min_cluster


@jit(
    nopython=True,
)
def agglomerative_clustering(clusters, number_of_clusters):
    while len(clusters) > number_of_clusters:
        print("Number of clusters: ", len(clusters))
        # Find the two closest clusters
        min_i, min_j = find_closest_clusters(clusters)

        # Merge the two clusters
        clusters[min_i] = merge_clusters(clusters[min_i], clusters[min_j])

        # Remove the merged cluster
        new_clusters = np.empty(
            (len(clusters) - 1, clusters.shape[1]), dtype=clusters.dtype
        )
        new_clusters[:min_j] = clusters[:min_j]
        new_clusters[min_j:] = clusters[min_j + 1 :]
        clusters = new_clusters

    return clusters


class Agglomerative:
    @staticmethod
    def agglomerative_segmentation(input_img, number_of_clusters):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Luv)
        segmentedImage = np.zeros(input_img.shape, dtype=np.uint8)

        # Initialize the clusters
        clusters = input_img.reshape(-1, 3)

        print("Start agglomerative clustering")
        # Perform agglomerative clustering based on stopping criteria
        clusters = agglomerative_clustering(clusters, number_of_clusters)

        print("Start assigning labels")
        # Assign labels to the pixels
        labels = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.int32)
        for i in prange(input_img.shape[0]):
            for j in prange(input_img.shape[1]):
                labels[i, j] = find_nearest_cluster(input_img[i, j], clusters)

        # Visualize the segmentation
        for i in prange(input_img.shape[0]):
            for j in prange(input_img.shape[1]):
                color = clusters[labels[i, j]]
                segmentedImage[i, j] = color

        segmentedImage = cv2.cvtColor(segmentedImage, cv2.COLOR_Luv2BGR)
        return segmentedImage


# Example usage
input_img = cv2.imread("test.jpg")
segmented_img = Agglomerative.agglomerative_segmentation(
    input_img, 5
)  # specify number of clusters
cv2.imshow("Segmented Image", segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
