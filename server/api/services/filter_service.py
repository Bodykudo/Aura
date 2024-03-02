import numpy as np
import cv2

from api.utils import convolve, get_image_dimensions, pad_image, read_image


class Filter:
    @staticmethod
    def apply_avg_filter(image_path, kernel_size):
        image = read_image(image_path)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
            kernel_size * kernel_size
        )
        result = convolve(image, kernel)
        return result

    @staticmethod
    def apply_gaussian_filter(image_path, kernel_size, sigma):
        kernel = Filter.gaussian_kernel(kernel_size, sigma)
        image = read_image(image_path)
        result = convolve(image, kernel)
        return result

    @staticmethod
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (size, size),
        )
        kernel = (kernel + kernel.T) / 2  # Ensure the kernel is symmetric
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_median_filter(image_path, kernel_size):
        image = read_image(image_path)
        height, width, channels = get_image_dimensions(image)
        padded_image, pad = pad_image(image, kernel_size)

        result = np.zeros_like(image)

        for i in range(channels):
            for y in range(pad, height + pad):
                for x in range(pad, width + pad):
                    if channels == 1:
                        result[y - pad, x - pad] = np.median(
                            padded_image[y - pad : y + pad + 1, x - pad : x + pad + 1]
                        )
                    else:
                        result[y - pad, x - pad, i] = np.median(
                            padded_image[
                                y - pad : y + pad + 1, x - pad : x + pad + 1, i
                            ]
                        )

        return result
