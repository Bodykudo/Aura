import numpy as np
import cv2

from api.utils import convolve, get_image_dimensions, pad_image, read_image

# from playground.mixer import Mixer


class Filter:
    @staticmethod
    def average_filter(image_path, kernel_size):
        image = read_image(image_path)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
            kernel_size * kernel_size
        )
        result = cv2.filter2D(image, -1, kernel)
        return result

    @staticmethod
    def gaussian_filter(image_path, kernel_size, sigma):
        kernel = Filter.gaussian_kernel(kernel_size, sigma)
        image = read_image(image_path)
        result = cv2.filter2D(image, -1, kernel)
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
        kernel = (kernel + kernel.T) / 2
        return kernel / np.sum(kernel)

    @staticmethod
    def median_filter(image_path, kernel_size):
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

    # @staticmethod
    # def apply_low_pass(image_path,radius):
    #     image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    #     width,height,_=get_image_dimensions(image)
    #     fft_image=Mixer.compute_fft(image)
    #     center_w,center_h=width//2,height//2
    #     filter_mask=np.zeros((width,height))
    #     filter_mask[center_w-radius:center_w+radius+1,center_h-radius:center_h+radius+1]=1
    #     smoothed_image=fft_image * filter_mask
    #     result = np.abs(np.fft.ifft2(np.fft.ifftshift(smoothed_image)))
    #     return result

    # @staticmethod
    # def apply_high_pass(image_path, radius):
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     width, height, _ = get_image_dimensions(image)
    #     fft_image = Mixer.compute_fft(image)
    #     center_w, center_h = width // 2, height // 2
    #     filter_mask = np.ones((width, height))
    #     filter_mask[center_w - radius:center_w + radius + 1, center_h - radius:center_h + radius + 1] = 0
    #     smoothed_image = fft_image * filter_mask
    #     result = np.abs(np.fft.ifft2(np.fft.ifftshift(smoothed_image)))
    #     return result
