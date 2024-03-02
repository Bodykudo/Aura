import cv2
import os
import numpy as np
from scipy.ndimage import convolve


class Edge_Detection:
    
    def __init__(self):
        pass
    
    def gray_scale(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    
    def gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    
    
    def gaussian_blur(self, image, kernel_size, sigma):
        kernel_size = int(kernel_size)
        kernel = self.gaussian_kernel(kernel_size, sigma)
        blurred_image = convolve(image, kernel)
        return blurred_image
    
    
    def sobel_filter(self, image):
        # image = Grayscale(GaussianBlur(image, 3, 1.0))
        image = self.gaussian_blur(self.gray_scale(image), 3, 1.0)
        G = np.zeros(image.shape)
        G_x = np.zeros(image.shape)
        G_y = np.zeros(image.shape)
        size = image.shape
        kernel_x = np.array(([-1, 0, 1], 
                             [-2, 0, 2], 
                             [-1, 0, 1]))
        
        kernel_y = np.array(([-1, -2, -1], 
                             [0, 0, 0], 
                             [1, 2, 1]))
        
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
                G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
        
        G = np.sqrt(np.square(G_x) + np.square(G_y))
        G = np.multiply(G, 255.0 / G.max())

        phase = np.rad2deg(np.arctan2(G_y, G_x))
        phase[phase < 0] += 180
        G = G.astype('uint8')
        return G, phase
    
    
    def non_maximum_suppression(self, image, angles):
        size = image.shape
        suppressed = np.zeros(size)
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    value_to_compare = max(image[i, j - 1], image[i, j + 1])
                elif (22.5 <= angles[i, j] < 67.5):
                    value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
                elif (67.5 <= angles[i, j] < 112.5):
                    value_to_compare = max(image[i - 1, j], image[i + 1, j])
                else:
                    value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
                
                if image[i, j] >= value_to_compare:
                    suppressed[i, j] = image[i, j]
                    
        suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
        return suppressed
    
    def double_threshold_hysteresis(self, image, low, high):
        weak = 50
        strong = 255
        size = image.shape
        result = np.zeros(size)
        weak_x, weak_y = np.where((image > low) & (image <= high))
        strong_x, strong_y = np.where(image >= high)
        result[strong_x, strong_y] = strong
        result[weak_x, weak_y] = weak
        dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
        dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
        size = image.shape
        
        while len(strong_x):
            x = strong_x[0]
            y = strong_y[0]
            strong_x = np.delete(strong_x, 0)
            strong_y = np.delete(strong_y, 0)
            for direction in range(len(dx)):
                new_x = x + dx[direction]
                new_y = y + dy[direction]
                if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                    result[new_x, new_y] = strong
                    np.append(strong_x, new_x)
                    np.append(strong_y, new_y)
        result[result != strong] = 0
        return result

    
    
    def Canny(self, image, low, high):
        image, angles = self.sobel_filter(image)
        image = self.non_maximum_suppression(image, angles)
        gradient = np.copy(image)
        image = self.double_threshold_hysteresis(image, low, high)
        return image, gradient
    
    
if __name__ == "__main__":
    image = cv2.imread('.\playground\cat.png')
    image2 = cv2.imread('.\playground\image1.jpg')
    image3 = cv2.imread('.\playground\image2.jpg')
    Edge = Edge_Detection()
    image, gradient = Edge.Canny(image, 0, 50)
    image2, gradient2 = Edge.Canny(image2, 0, 50)
    image3, gradient3 = Edge.Canny(image3, 0, 30)
    cv2.imshow('image', image)
    cv2.imshow('image2', image2)
    cv2.imshow('image3', image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()