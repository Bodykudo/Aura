import cv2
import numpy as np

from api.utils import read_image


class Thresholding:

    @staticmethod
    def preprocess(image):
        grayscale_image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        )
        histogram = np.histogram(grayscale_image.flatten(), bins=256, range=[0, 256])[0]
        cumulative_histogram = histogram.cumsum()
        global_mean_intensity = np.sum(np.arange(256) * histogram) / histogram.sum()
        return grayscale_image, histogram, cumulative_histogram, global_mean_intensity

    @staticmethod
    def find_spectral_thresholds(histogram: np.ndarray, global_mean_intensity: float):
        max_variance = 0
        for high_threshold in range(1, 256):
            for low_threshold in range(1, high_threshold):
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
                        / (weights[0] + 1e-6),
                        np.dot(
                            np.arange(low_threshold, high_threshold),
                            histogram[low_threshold:high_threshold],
                        )
                        / (weights[1] + 1e-6),
                        np.dot(
                            np.arange(high_threshold, 256), histogram[high_threshold:]
                        )
                        / (weights[2] + 1e-6),
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
            image = read_image(image_or_path, grayscale=True)  # Global default
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
    def local_thresholding(
        image_path: str, thresholding_type: str, window_size: int, offset: int
    ):
        image = read_image(image_path, grayscale=True)
        thresholded_image = np.zeros(image.shape[:2], dtype=np.uint8)
        image_height, image_width = image.shape[:2]
        window = (window_size, window_size)
        step_y, step_x = window

        for y in range(0, image_height, step_y):
            for x in range(0, image_width, step_x):
                sub_image = image[
                    y : min(y + step_y, image_height), x : min(x + step_x, image_width)
                ]
                if thresholding_type == "optimal":
                    _, threshold = Thresholding.optimal_thresholding(sub_image)
                    threshold = threshold - offset
                    print("threshold", threshold)
                    local_thresholded = (sub_image > threshold).astype(np.uint8) * 255

                elif thresholding_type == "otsu":
                    _, threshold = Thresholding.otsu_thresholding(sub_image)
                    threshold = threshold - offset
                    print("threshold", threshold)
                    local_thresholded = (sub_image > threshold).astype(np.uint8) * 255

                elif thresholding_type == "spectral":
                    local_thresholded = Thresholding.spectral_thresholding(sub_image)
                    print("threshold", local_thresholded.shape)

                thresholded_image[
                    y : min(y + step_y, image_height), x : min(x + step_x, image_width)
                ] = local_thresholded

        return thresholded_image

    @staticmethod
    def optimal_thresholding(image_or_path):
        if isinstance(image_or_path, str):
            image = read_image(image_or_path, grayscale=True)
        else:
            image = image_or_path

        height, width = image.shape[:2]
        corners = [
            image[0, 0],
            image[0, width - 1],
            image[height - 1, 0],
            image[height - 1, width - 1],
        ]
        threshold = np.mean(corners)
        while True:
            class1_mean = np.mean(image[image < threshold])
            class2_mean = np.mean(image[image >= threshold])
            new_threshold = (class1_mean + class2_mean) / 2
            if np.abs(new_threshold - threshold) < 1e-6:
                break
            threshold = new_threshold

        binary_image = (image > threshold).astype(np.uint8) * 255

        return binary_image, threshold

    @staticmethod
    def otsu_thresholding(image_or_path):
        if isinstance(image_or_path, str):
            image = read_image(image_or_path, grayscale=True)
        else:
            image = image_or_path

        grayscale_image, histogram, _, global_mean_intensity = Thresholding.preprocess(
            image
        )

        pdf = histogram / float(np.sum(histogram))
        cdf = np.cumsum(pdf)
        cumulative_sum_intensity = np.cumsum(np.arange(256) * pdf)

        background_weight = cdf
        foreground_weight = 1.0 - background_weight

        class1_mean = cumulative_sum_intensity / (background_weight + 1e-6)
        class2_mean = (global_mean_intensity - cumulative_sum_intensity) / (
            foreground_weight + 1e-6
        )

        inter_class_variances = (
            background_weight * foreground_weight * (class1_mean - class2_mean) ** 2
        )

        threshold = np.argmax(inter_class_variances)

        binary_image = (image > threshold).astype(np.uint8) * 255

        return binary_image, threshold
