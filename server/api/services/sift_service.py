import cv2
import numpy as np

from api.utils import read_image


class SIFTDetector:
    @staticmethod
    def build_scale_space_pyramid(
        image_path: str,
        num_octaves: int = 3,
        num_scales: int = 5,
        sigma: float = 1.6,
        downsampling_factor: float = 2,
    ):

        pyramid = []
        image = read_image(image_path)
        current_image = image.copy()
        for octave_level in range(num_octaves):
            octave = []
            for scale_level in range(num_scales):
                blurred_image = cv2.GaussianBlur(current_image, (0, 0), sigma)
                octave.append(blurred_image)
                sigma *= 2.0

            new_width = int(current_image.shape[1] / downsampling_factor)
            new_height = int(current_image.shape[0] / downsampling_factor)
            current_image = cv2.resize(
                current_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            )
            sigma = 1.6
            pyramid.append(octave)

        return pyramid

    @staticmethod
    def generate_DoG_pyramid(gaussian_pyramid):
        DoG_pyramid = []
        for octave in gaussian_pyramid:
            DoG_octave = []
            for i in range(len(octave) - 1):
                DoG_image = octave[i + 1] - octave[i]
                DoG_octave.append(DoG_image)
            DoG_pyramid.append(DoG_octave)

        return DoG_pyramid
