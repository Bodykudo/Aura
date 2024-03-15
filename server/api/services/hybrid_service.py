import numpy as np
import cv2

from api.utils import get_image_dimensions, compute_fft, read_image


class Hybrid:

    @staticmethod
    def img_adjustment(image1, image2):
        width_1, height_1, _ = get_image_dimensions(image1)
        width_2, height_2, _ = get_image_dimensions(image2)
        if width_1 > width_2:
            width = width_2
        else:
            width = width_1

        if height_1 > height_2:
            height = height_2
        else:
            height = height_1
        adjusted_img1 = cv2.resize(image1, (width, height))
        adjusted_img2 = cv2.resize(image2, (width, height))
        return adjusted_img1, adjusted_img2

    @staticmethod
    def apply_low_pass(image, radius):
        width, height, _ = get_image_dimensions(image)
        fft_image = compute_fft(image)
        center_w, center_h = width // 2, height // 2
        filter_mask = np.zeros((width, height))
        filter_mask[
            center_w - radius : center_w + radius + 1,
            center_h - radius : center_h + radius + 1,
        ] = 1
        masked_fft_img = fft_image * filter_mask
        filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_fft_img)))
        return masked_fft_img, filtered_image

    @staticmethod
    def apply_high_pass(image, radius):
        width, height, _ = get_image_dimensions(image)
        fft_image = compute_fft(image)
        center_w, center_h = width // 2, height // 2
        filter_mask = np.ones((width, height))
        filter_mask[
            center_w - radius : center_w + radius + 1,
            center_h - radius : center_h + radius + 1,
        ] = 0
        masked_fft_image = fft_image * filter_mask
        filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_fft_image)))
        return masked_fft_image, filtered_image

    @staticmethod
    def apply_mixer(img_1_path, img_2_path, filter_1, radius):
        # Read image
        img_1 = read_image(img_1_path, grayscale=True)
        img_2 = read_image(img_2_path, grayscale=True)
        adjusted_img1, adjusted_img2 = Hybrid.img_adjustment(img_1, img_2)
        if filter_1 == "high":
            fft_img1, output_img1 = Hybrid.apply_high_pass(adjusted_img1, radius)
            fft_img2, output_img2 = Hybrid.apply_low_pass(adjusted_img2, radius)
        else:
            fft_img2, output_img2 = Hybrid.apply_high_pass(adjusted_img2, radius)
            fft_img1, output_img1 = Hybrid.apply_low_pass(adjusted_img1, radius)

        mixed_img = fft_img1 + fft_img2
        output_mixed_img = np.abs(np.fft.ifft2(np.fft.ifftshift(mixed_img)))
        return output_mixed_img, output_img1, output_img2
