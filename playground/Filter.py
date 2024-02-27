import numpy as np
import cv2
from scipy.ndimage import convolve


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
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)


    @staticmethod
    def apply_gaussian_filter(image_path, kernel):

            original_image = cv2.imread(image_path)
            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_height, image_width, channels = image.shape
            kernel_height, kernel_width = kernel.shape

            # Compute the padding needed for the convolution
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2

            # Pad the image with zeros
            padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

            # Initialize the filtered image with zeros
            filtered_image = np.zeros_like(image)

            # Perform the convolution for each channel
            for c in range(channels):
                for i in range(pad_height, image_height + pad_height):
                    for j in range(pad_width, image_width + pad_width):
                        # Extract the region from the padded image
                        region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1, c]

                        # Compute the element-wise multiplication with the kernel
                        convolution_result = np.sum(region * kernel)

                        # Assign the result to the corresponding pixel in the filtered image
                        filtered_image[i - pad_height, j - pad_width, c] = convolution_result
            return filtered_image


    @staticmethod
    def apply_median_filter(image_path,kernel_size):
        original_image = cv2.imread(image_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        output_image = np.zeros_like(image)

        half_kernel = kernel_size // 2

        for i in range(half_kernel, image.shape[0] - half_kernel):
            for j in range(half_kernel, image.shape[1] - half_kernel):
                red_values = []
                green_values = []
                blue_values = []

                for m in range(-half_kernel, half_kernel + 1):
                    for n in range(-half_kernel, half_kernel + 1):
                        pixel = image[i + m, j + n]
                        blue_values.append(pixel[0])
                        green_values.append(pixel[1])
                        red_values.append(pixel[2])

                    # Sort the values and select the median
                    red_values.sort()
                    green_values.sort()
                    blue_values.sort()

                    median_index = len(red_values) // 2
                    median_pixel = (blue_values[median_index], green_values[median_index], red_values[median_index])

                    # Set the pixel in the destination image to the median value
                    output_image[i, j] = median_pixel

        return output_image

