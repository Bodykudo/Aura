import numpy as np

from api.utils import read_image


class Thresholding:
    @staticmethod
    def global_thresholding(image_path, threshold: int = 127):
        image = read_image(image_path, grayscale=True)
        return np.where(image > threshold, 255, 0).astype(np.uint8)

    @staticmethod
    def local_thresholding(image_path, block_size: int):
        image = read_image(image_path, grayscale=True)
        rows, cols = np.meshgrid(
            np.arange(0, image.shape[0] - block_size + 1, block_size),
            np.arange(0, image.shape[1] - block_size + 1, block_size),
            indexing="ij",
        )
        start_indices = (
            np.array(
                np.meshgrid(
                    np.arange(0, image.shape[0] - block_size + 1, block_size),
                    np.arange(0, image.shape[1] - block_size + 1, block_size),
                    indexing="ij",
                )
            )
            .reshape(2, -1)
            .T
        )

        blocks = [
            image[r : r + block_size, c : c + block_size] for r, c in start_indices
        ]
        thresholds = np.mean(blocks, axis=(1, 2))

        local_threshold = np.zeros_like(image)
        for (r, c), block, threshold in zip(start_indices, blocks, thresholds):
            local_threshold[r : r + block_size, c : c + block_size] = np.where(
                block > threshold, 255, 0
            )

        return local_threshold.astype(np.uint8)
