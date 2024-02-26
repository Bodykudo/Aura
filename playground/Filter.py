import numpy as np
import cv2


class Filter:
    @staticmethod
    def apply_avg_filter(image,kernel_size):

        # Read Image and convert it to grayscale

        original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        filtered_image = original_image.copy()
        rows, cols = original_image.shape

        # Check that Kernel size is ODD

        if kernel_size % 2 == 0:
            kernel_size += 1

        half_kernel = kernel_size // 2

        for i in range(half_kernel, rows - half_kernel):
            for j in range(half_kernel, cols - half_kernel):
                neighborhood = original_image[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]
                average_value = np.mean(neighborhood)
                filtered_image[i, j] = int(average_value)

        return filtered_image

    @staticmethod
    def apply_gaussian_filter(image):
        pass

    @staticmethod
    def apply_median_filter(image):
        pass

