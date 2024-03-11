import numpy as np
import cv2

from api.utils import get_image_dimensions, compute_fft


class Hybrid:
    @staticmethod
    def apply_low_pass(image_path, radius):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        width, height, _ = get_image_dimensions(image)
        fft_image = compute_fft(image)
        center_w, center_h = width // 2, height // 2
        filter_mask = np.zeros((width, height))
        filter_mask[
            center_w - radius : center_w + radius + 1,
            center_h - radius : center_h + radius + 1,
        ] = 1
        masked_fft_img = fft_image * filter_mask
        return masked_fft_img

    @staticmethod
    def apply_high_pass(image_path, radius):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        width, height, _ = get_image_dimensions(image)
        fft_image = compute_fft(image)
        center_w, center_h = width // 2, height // 2
        filter_mask = np.ones((width, height))
        filter_mask[
            center_w - radius : center_w + radius + 1,
            center_h - radius : center_h + radius + 1,
        ] = 0
        masked_fft_image = fft_image * filter_mask
        return masked_fft_image

    @staticmethod
    def apply_mixer(img_1_path, img_2_path, filter_1, radius):
        if filter_1 == "High Pass Filter":
            fft_img1 = Hybrid.apply_high_pass(img_1_path, radius)
            output_img1 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img1)))
            fft_img2 = Hybrid.apply_low_pass(img_2_path, radius)
            output_img2 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img2)))
        else:
            fft_img2 = Hybrid.apply_high_pass(img_2_path, radius)
            output_img2 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img2)))
            fft_img1 = Hybrid.apply_low_pass(img_1_path, radius)
            output_img1 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img1)))

        mixed_img = fft_img1 + fft_img2
        output_mixed_img = np.abs(np.fft.ifft2(np.fft.ifftshift(mixed_img)))
        return output_mixed_img, output_img1, output_img2
