import numpy as np
import cv2

from api.utils import (
    gaussian_kernel,
    get_image_dimensions,
    pad_image,
    read_image,
)


class Filter:
    @staticmethod
    def average_filter(image_path: str, kernel_size: int):
        image = read_image(image_path)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
            kernel_size * kernel_size
        )
        filtered_image = cv2.filter2D(image, -1, kernel)
        return filtered_image

    @staticmethod
    def gaussian_filter(image_path: str, kernel_size: int, sigma: float):
        kernel = gaussian_kernel(kernel_size, sigma)
        image = read_image(image_path)
        filtered_image = cv2.filter2D(image, -1, kernel)
        return filtered_image

    @staticmethod
    def median_filter(image_path: str, kernel_size: int):
        image = read_image(image_path)
        height, width, channels = get_image_dimensions(image)
        padded_image, pad = pad_image(image, kernel_size)

        filtered_image = np.zeros_like(image)

        for i in range(channels):
            for y in range(pad, height + pad):
                for x in range(pad, width + pad):
                    if channels == 1:
                        filtered_image[y - pad, x - pad] = np.median(
                            padded_image[y - pad : y + pad + 1, x - pad : x + pad + 1]
                        )
                    else:
                        filtered_image[y - pad, x - pad, i] = np.median(
                            padded_image[
                                y - pad : y + pad + 1, x - pad : x + pad + 1, i
                            ]
                        )

        return filtered_image
