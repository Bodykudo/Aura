import cv2
import numpy as np
from numba import prange

from api.utils import (
    agglomerative_clustering,
    find_nearest_cluster,
    compute_distances,
    euclidean_distance,
    read_image,
)


class Segmentation:
    @staticmethod
    def kmeans_segmentation(image_path: str, K: int, max_iterations: int):
        image = read_image(image_path)
        transformed_image = image.reshape((-1, 3))
        transformed_image = np.float32(transformed_image)

        num_pixels, _ = transformed_image.shape
        random_indices = np.random.choice(num_pixels, K, replace=False)
        centroids = transformed_image[random_indices]
        for _ in range(max_iterations):
            distances = np.sqrt(
                ((transformed_image - centroids[:, np.newaxis]) ** 2).sum(axis=2)
            )
            closest_centroids = np.argmin(distances, axis=0)
            clusters = {i: np.where(closest_centroids == i)[0] for i in range(K)}
            new_centroids = np.array(
                [
                    transformed_image[cluster].mean(axis=0)
                    for cluster in clusters.values()
                ]
            )

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        labels = np.empty(num_pixels)
        for cluster_index, pixel_indices in clusters.items():
            labels[pixel_indices] = cluster_index

        labels = labels.astype(int)
        segmented_image = labels.reshape(image.shape[:-1])
        masked_image = np.zeros(image.shape, dtype=np.uint8)
        for i in range(K):
            masked_image[segmented_image == i] = centroids[i]
        masked_image = np.uint8(masked_image)
        return masked_image

    @staticmethod
    def mean_shift_segmentation(image_path: str, window_size: int, threshold: float):
        input_image = read_image(image_path)
        image_height, image_width, _ = input_image.shape
        segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        feature_space = np.zeros((image_height * image_width, 5))

        pixel_counter = 0
        for row in range(image_height):
            for col in range(image_width):
                feature_space[pixel_counter] = [
                    input_image[row][col][0],
                    input_image[row][col][1],
                    input_image[row][col][2],
                    row,
                    col,
                ]
                pixel_counter += 1

        is_new_mean_random = True
        current_mean_array = np.zeros((1, 5))
        while len(feature_space) > 0:
            if is_new_mean_random:
                current_mean_index = np.random.randint(0, feature_space.shape[0])
                current_mean_array[0] = feature_space[current_mean_index]

            pixel_distances = compute_distances(feature_space, current_mean_array)
            pixels_within_window = np.where(pixel_distances < window_size)[0]

            mean_color = np.mean(feature_space[pixels_within_window, :3], axis=0)
            mean_position = np.mean(feature_space[pixels_within_window, 3:], axis=0)
            color_distance_to_mean = euclidean_distance(
                mean_color, current_mean_array[0][:3]
            )
            position_distance_to_mean = euclidean_distance(
                mean_position, current_mean_array[0][3:]
            )
            total_distance_to_mean = color_distance_to_mean + position_distance_to_mean

            if total_distance_to_mean < threshold:
                new_color_array = np.zeros((1, 3))
                new_color_array[0] = mean_color
                is_new_mean_random = True
                segmented_image[
                    feature_space[pixels_within_window, 3].astype(int),
                    feature_space[pixels_within_window, 4].astype(int),
                ] = new_color_array
                feature_space[pixels_within_window, :] = -1
                feature_space = feature_space[feature_space[:, 0] != -1]

            else:
                is_new_mean_random = False
                current_mean_array[0, :3] = mean_color
                current_mean_array[0, 3:] = mean_position
        return cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def get_8_connected(x, y, shape):
        xmax = shape[0] - 1
        ymax = shape[1] - 1

        connected_pixels = []

        for dx in range(3):
            for dy in range(3):
                connected_pixel_x = x + dx - 1
                connected_pixel_y = y + dy - 1
                if (
                    (0 <= connected_pixel_x <= xmax)
                    and (0 <= connected_pixel_y <= ymax)
                    and not (connected_pixel_x == x and connected_pixel_y == y)
                ):
                    connected_pixels.append((connected_pixel_x, connected_pixel_y))

        return connected_pixels

    @staticmethod
    def region_growing_segmentaion(
        image_path: str, thershold: int, seed_points: list[tuple]
    ):
        print(seed_points)
        original_image = read_image(image_path)
        image_gray = read_image(image_path, grayscale=True)
        _, img = cv2.threshold(image_gray, thershold, 255, cv2.THRESH_BINARY)
        processed = np.full((img.shape[0], img.shape[1]), False)

        outimg = np.zeros_like(img)

        for index, pix in enumerate(seed_points):
            processed[pix[0], pix[1]] = True
            outimg[pix[0], pix[1]] = img[pix[0], pix[1]]

        while len(seed_points) > 0:
            pix = seed_points[0]

            for coord in Segmentation.get_8_connected(pix[0], pix[1], img.shape):
                if not processed[coord[0], coord[1]]:
                    if img[coord[0], coord[1]] != 0:
                        outimg[coord[0], coord[1]] = outimg[pix[0], pix[1]]
                        if not processed[coord[0], coord[1]]:
                            seed_points.append(coord)
                        processed[coord[0], coord[1]] = True

            seed_points.pop(0)

        contours, _ = cv2.findContours(
            processed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_with_contours = original_image.copy()
        for contour in contours:
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

        return image_with_contours

    @staticmethod
    def agglomerative_segmentation(image_path: str, number_of_clusters: int):
        image = read_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        segmentedImage = np.zeros(image.shape, dtype=np.uint8)

        clusters = image.reshape(-1, 3)

        clusters = agglomerative_clustering(clusters, number_of_clusters)

        # Assign labels to the pixels
        labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        for i in prange(image.shape[0]):
            for j in prange(image.shape[1]):
                labels[i, j] = find_nearest_cluster(image[i, j], clusters)

        for i in prange(image.shape[0]):
            for j in prange(image.shape[1]):
                color = clusters[labels[i, j]]
                segmentedImage[i, j] = color

        segmentedImage = cv2.cvtColor(segmentedImage, cv2.COLOR_BGR2RGB)
        return segmentedImage
