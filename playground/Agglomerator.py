import numpy as np
import cv2

class Agglomerative:
    @staticmethod
    def calculate_distance(a, b):
        return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2) + np.power(a[2] - b[2], 2))

    @staticmethod
    def merge_clusters(a, b):
        r = (a[0] + b[0]) // 2
        g = (a[1] + b[1]) // 2
        b = (a[2] + b[2]) // 2
        return (r, g, b)

    @staticmethod
    def agglomerative_segmentation(input_img, number_of_clusters):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Luv)
        segmentedImage = np.zeros(input_img.shape, dtype=np.uint8)

        # Initialize the clusters
        clusters = []

        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                clusters.append(input_img[i, j])

        # Perform agglomerative clustering based on stopping criteria
        while len(clusters) > number_of_clusters:
            # Find the two closest clusters
            min_dist = float('inf')
            min_i, min_j = 0, 0
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = Agglomerative.calculate_distance(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j

            # Merge the two clusters
            clusters[min_i] = Agglomerative.merge_clusters(clusters[min_i], clusters[min_j])
            clusters.pop(min_j)

        # Assign labels to the pixels
        labels = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.int32)
        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                min_cluster = 0
                min_dist = float('inf')
                for k in range(len(clusters)):
                    dist = Agglomerative.calculate_distance(input_img[i, j], clusters[k])
                    if dist < min_dist:
                        min_dist = dist
                        min_cluster = k
                labels[i, j] = min_cluster

        # Visualize the segmentation
        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                color = clusters[labels[i, j]]
                segmentedImage[i, j] = color

        segmentedImage = cv2.cvtColor(segmentedImage, cv2.COLOR_Luv2BGR)
        return segmentedImage

# Example usage
input_img = cv2.imread('input_image.jpg')
segmented_img = Agglomerative.agglomerative_segmentation(input_img, 5)  # specify number of clusters
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()