# from server.api.utils import read_image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from api.services.filter_service import Filter

class Mixer():
    @staticmethod
    def apply_mixer(img_1_path,img_2_path,filter_1,radius):
        if filter_1=='High Pass Filter':
            fft_img1=Filter.apply_high_pass(img_1_path,radius)
            output_img1 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img1)))
            fft_img2=Filter.apply_low_pass(img_2_path,radius)
            output_img2= np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img2)))
        else:
            fft_img2 = Filter.apply_high_pass(img_2_path, radius)
            output_img2 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img2)))
            fft_img1 = Filter.apply_low_pass(img_1_path, radius)
            output_img1 = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img1)))

        mixed_img=fft_img1+fft_img2
        output_mixed_img= np.abs(np.fft.ifft2(np.fft.ifftshift(mixed_img)))
        return output_mixed_img,output_img1,output_img2







