import numpy as np
import cv2
import numpy as np
import random

from api.utils import (
    compute_distances,
    euclidean_distance,
    read_image,
    resize_image,
    gaussian_filter,

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
    def region_growing_segmentaion(image_path: str, threshold: int, neighbours_number: int):
        image = cv2.imread(image_path)
        if len(image.shape) == 3:  # If image is colored
            original_image = np.copy(image)  # Make a copy of the original image for background
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            original_image  = image  # No need for conversion if already grayscale

        image = resize_image(grayscale_image)
        image = gaussian_filter(image)
        segmentation = np.zeros(image.shape, dtype=np.uint8)
        image_height, image_width = image.shape
        threshold = threshold
        seeds = []

        if neighbours_number == 4:
            orientations = [(1,0),(0,1),(-1,0),(0,-1)]
        elif neighbours_number == 8:
            orientations = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        seeds.append((int(image_height/2), int(image_width/2)))
        seeds.append((int(2*image_height/3), int(2*image_width/3)))
        seeds.append((int(image_height/3-10), int(2*image_width/3)))

        for seed in seeds:
            curr_pixel = [seed[1], seed[0]]

            if segmentation[curr_pixel[0], curr_pixel[1]] == 255:
                continue

            contour = []
            seg_size = 1
            mean_seg_value = (image[curr_pixel[0], curr_pixel[1]])
            dist = 0

            while(dist < threshold):
                segmentation[curr_pixel[0], curr_pixel[1]] = 255
                contour = Segmentation.explore_neighbours(contour, curr_pixel, orientations, segmentation)
                nearest_neighbour_idx, dist = Segmentation.get_nearest_neighbour(contour, mean_seg_value, image)

                if nearest_neighbour_idx == -1:
                    break

                curr_pixel = contour[nearest_neighbour_idx]
                seg_size += 1
                mean_seg_value = (mean_seg_value * seg_size + float(image[curr_pixel[0], curr_pixel[1]])) / (seg_size + 1)
                del contour[nearest_neighbour_idx]
        # Generate random colors for each segmented region
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(256)]
        colored_image = np.zeros((image_height,image_width, 3), dtype=np.uint8)
        for y in range(image_height):
            for x in range(image_width):
                label = segmentation[y, x]
                if label != 0:
                    colored_image[y, x] = colors[label]
                else:
                    colored_image[y, x] = original_image[y, x]

        return colored_image
    
    @staticmethod
    def explore_neighbours(contour, current_pixel, orientations, segmentation):
        for orientation in orientations:
            neighbour = Segmentation.get_neighbouring_pixel(current_pixel, orientation, segmentation.shape)
            if neighbour is None:
                continue
            if segmentation[neighbour[0], neighbour[1]] == 0:
                contour.append(neighbour)
                segmentation[neighbour[0], neighbour[1]] = 150
        return contour

    @staticmethod
    def get_neighbouring_pixel(current_pixel, orient, img_shape):
        neighbour = ((current_pixel[0] + orient[0]), (current_pixel[1] + orient[1]))
        if Segmentation.is_pixel_inside_image(neighbour, img_shape):
            return neighbour
        else:
            return None

    @staticmethod
    def get_nearest_neighbour(contour, mean_seg_value, image):
        dist_list = [abs(int(image[pixel[0], pixel[1]]) - mean_seg_value) for pixel in contour]
        if len(dist_list) == 0:
            return -1, 1000
        min_dist = min(dist_list)
        index = dist_list.index(min_dist)
        return index, min_dist

    @staticmethod
    def is_pixel_inside_image(pixel, img_shape):
        return 0 <= pixel[0] < img_shape[0] and 0 <= pixel[1] < img_shape[1]