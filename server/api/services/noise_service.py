import cv2
import numpy as np

from api.utils import read_image


class Noise:
    @staticmethod
    def uniform_noise(image_path, noise_value=50):
        image = read_image(image_path)
        noise = np.random.uniform(-noise_value, noise_value, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def gaussian_noise(image_path, mean=0, variance=25):
        image = read_image(image_path)
        noise = np.random.normal(mean, variance, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def salt_and_pepper_noise(image_path, salt_prob=0.02, pepper_prob=0.02):
        image = read_image(image_path)
        noisy_image = np.copy(image)
        salt_pixels = np.random.random(image.shape) < salt_prob
        noisy_image[salt_pixels] = 255
        pepper_pixels = np.random.random(image.shape) < pepper_prob
        noisy_image[pepper_pixels] = 0
        return noisy_image
