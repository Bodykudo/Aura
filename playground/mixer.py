# from server.api.utils import read_image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from server.api.utils import read_image

class Mixer():
    @staticmethod
    def compute_fft(image):
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        # fft = (20 * np.log10(0.1 + f_transform_shifted))
        return f_transform_shifted

    @staticmethod
    def apply_mixer():
        pass
