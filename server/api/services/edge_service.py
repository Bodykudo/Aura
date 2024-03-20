import numpy as np
import cv2
import skimage.exposure as exposure

from api.utils import read_image


class EdgeDetector:
    @staticmethod
    def sobel_edge_detection(image_path: str, kernel_size: int, direction="both"):
        image = read_image(image_path, grayscale=True)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_x = cv2.filter2D(image, cv2.CV_32F, sobel_x_kernel)
        sobel_y = cv2.filter2D(image, cv2.CV_32F, sobel_y_kernel)

        if direction == "x":
            sobel_x_normalized = (
                exposure.rescale_intensity(
                    sobel_x, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_x_normalized
        elif direction == "y":
            sobel_y_normalized = (
                exposure.rescale_intensity(
                    sobel_y, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_y_normalized
        elif direction == "both":
            sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            sobel_magnitude = (
                exposure.rescale_intensity(
                    sobel_magnitude, in_range="image", out_range=(0, 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )

            return sobel_magnitude
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
