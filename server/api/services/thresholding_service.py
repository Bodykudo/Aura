import cv2
import numpy as np
import matplotlib.pyplot as plt

from api.utils import read_image


class Thresholding:
    @staticmethod
    def preprocess(image):
        grayscale_image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        )
        histogram = np.histogram(grayscale_image, bins=256, range=[0, 256])[0]
        cumulative_histogram = histogram.cumsum()
        global_mean_intensity = np.sum(np.arange(256) * histogram) / histogram.sum()
        return grayscale_image, histogram, cumulative_histogram, global_mean_intensity

    @staticmethod
    def find_spectral_thresholds(histogram: np.ndarray, global_mean_intensity: float):
        max_variance = 0
        for high_threshold in range(1, 256):
            for low_threshold in range(high_threshold):
                weights = np.array(
                    [
                        histogram[:low_threshold].sum(),
                        histogram[low_threshold:high_threshold].sum(),
                        histogram[high_threshold:].sum(),
                    ]
                )
                if weights[1] == 0:  # Skip if weight is 0 to avoid division by zero
                    continue
                means = np.array(
                    [
                        np.dot(np.arange(0, low_threshold), histogram[:low_threshold])
                        / weights[0],
                        np.dot(
                            np.arange(low_threshold, high_threshold),
                            histogram[low_threshold:high_threshold],
                        )
                        / weights[1],
                        np.dot(
                            np.arange(high_threshold, 256), histogram[high_threshold:]
                        )
                        / weights[2],
                    ]
                )
                variance = np.dot(weights, (means - global_mean_intensity) ** 2)
                if variance > max_variance:
                    max_variance = variance
                    optimal_low_threshold, optimal_high_threshold = (
                        low_threshold,
                        high_threshold,
                    )

        return optimal_low_threshold, optimal_high_threshold

    @staticmethod
    def apply_threshold(image, low_threshold: int, high_threshold: int):

        binary_image = np.zeros_like(image)
        binary_image[image < low_threshold] = 0
        binary_image[(image >= low_threshold) & (image < high_threshold)] = 128
        binary_image[image >= high_threshold] = 255
        return binary_image

    @staticmethod
    def spectral_thresholding(image_or_path):
        if isinstance(image_or_path, str):
            image = read_image(image_or_path)
        else:
            image = image_or_path

        grayscale_image, histogram, _, global_mean_intensity = Thresholding.preprocess(
            image
        )
        optimal_low_threshold, optimal_high_threshold = (
            Thresholding.find_spectral_thresholds(histogram, global_mean_intensity)
        )
        binary_image = Thresholding.apply_threshold(
            grayscale_image, optimal_low_threshold, optimal_high_threshold
        )
        return binary_image

    @staticmethod
    def spectral_thresholding_local(image_path: str, window_size: int):
        image = read_image(image_path)
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        image_height, image_width = image.shape[:2]
        window = (window_size, window_size)
        step_y, step_x = window

        for y in range(0, image_height, step_y):
            for x in range(0, image_width, step_x):
                sub_image = image[
                    y : min(y + step_y, image_height), x : min(x + step_x, image_width)
                ]
                local_thresholded = Thresholding.spectral_thresholding(sub_image)

                result[
                    y : min(y + step_y, image_height), x : min(x + step_x, image_width)
                ] = local_thresholded

        return result
