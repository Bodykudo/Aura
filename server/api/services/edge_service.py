import numpy as np
import cv2
import skimage.exposure as exposure

from api.utils import read_image


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif 22.5 <= angles[i, j] < 67.5:
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif 67.5 <= angles[i, j] < 112.5:
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]

    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if (new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (
                result[new_x, new_y] == weak
            ):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


class EdgeDetector:
    @staticmethod
    def sobel_edge_detection(
        image_path: str, kernel_size: int, direction="both", sigma: int = 0
    ):
        image = read_image(image_path, grayscale=True)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_x = cv2.filter2D(image, cv2.CV_32F, sobel_x_kernel)
        sobel_y = cv2.filter2D(image, cv2.CV_32F, sobel_y_kernel)

        phase = np.rad2deg(np.arctan2(sobel_y, sobel_x))
        phase[phase < 0] += 180

        if direction == "x":
            sobel_x_normalized = (
                exposure.rescale_intensity(
                    sobel_x, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_x_normalized, phase
        elif direction == "y":
            sobel_y_normalized = (
                exposure.rescale_intensity(
                    sobel_y, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_y_normalized, phase
        elif direction == "both":
            sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            sobel_magnitude = (
                exposure.rescale_intensity(
                    sobel_magnitude, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            return sobel_magnitude, phase
        else:
            raise ValueError("Invalid direction. Please use x, y or both.")

    @staticmethod
    def prewitt_edge_detection(image_path: str, kernel_size: int):
        image = read_image(image_path, grayscale=True)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_y)
        magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
        prewitt_magnitude = (
            exposure.rescale_intensity(magnitude, in_range="image", out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        return prewitt_magnitude

    @staticmethod
    def roberts_edge_detection(image_path: str, kernel_size: int):
        image = read_image(image_path, grayscale=True)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        roberts_kernel_x = np.array([[1, 0], [0, -1]])
        roberts_kernel_y = np.array([[0, 1], [-1, 0]])
        roberts_x = cv2.filter2D(image, -1, roberts_kernel_x)
        roberts_y = cv2.filter2D(image, -1, roberts_kernel_y)
        roberts_magnitude = np.abs(roberts_x) + np.abs(roberts_y)
        roberts_magnitude = np.clip(roberts_magnitude, 0, 255)
        return roberts_magnitude.astype(np.uint8)

    @staticmethod
    def canny_edge_detection(
        image_path: str,
        kernel_size: int,
        sigma: float,
        low_threshold: int,
        high_threshold: int,
    ):
        if low_threshold > high_threshold:
            raise ValueError("Low threshold should be less than high threshold.")

        image = read_image(image_path, grayscale=True)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges

    @staticmethod
    def canny_edge_detection_new(
        image_path: str,
        kernel_size: int,
        sigma: float,
        low_threshold: int,
        high_threshold: int,
    ):
        if low_threshold > high_threshold:
            raise ValueError("Low threshold should be less than high threshold.")

        image, angles = EdgeDetector.sobel_edge_detection(
            image_path, kernel_size, "both", sigma
        )
        image = non_maximum_suppression(image, angles)
        image = double_threshold_hysteresis(image, low_threshold, high_threshold)
        return image
