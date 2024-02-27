import numpy as np
import cv2


class Filter:
    @staticmethod
    def apply_avg_filter(image_path,kernel_size):

        original_image = cv2.imread(image_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        result_image = np.zeros_like(image, dtype=np.uint8)


        for i in range(height):
            for j in range(width):
                neighborhood = image[max(0, i - kernel_size // 2):min(height, i + kernel_size // 2 + 1),
                               max(0, j - kernel_size // 2):min(width, j + kernel_size // 2 + 1)]


                average_value = np.mean(neighborhood, axis=(0, 1))


                result_image[i, j] = np.round(average_value).astype(np.uint8)


        return result_image

    @staticmethod
    def apply_gaussian_filter(image):
        pass

    @staticmethod
    def apply_median_filter(image):
        pass

