import numpy as np

from api.utils import read_image


class Thresholding:
    @staticmethod
    def global_thresholding(image_path: str, threshold: int = 127):
        image = read_image(image_path, grayscale=True)
        return np.where(image > threshold, 255, 0).astype(np.uint8)

    @staticmethod
    def local_thresholding(
        image_path: str,
        threshold_margin: int,
        block_size: int,
    ):
        window_size = (block_size, block_size)
        image = read_image(image_path, grayscale=True)
        height, width = image.shape
        output = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                window = image[
                    max(0, i - window_size[0] // 2) : min(
                        height, i + window_size[0] // 2 + 1
                    ),
                    max(0, j - window_size[1] // 2) : min(
                        width, j + window_size[1] // 2 + 1
                    ),
                ]
                local_mean = np.mean(window.flatten())
                output[i, j] = 255 if image[i, j] > local_mean - threshold_margin else 0
        return output
