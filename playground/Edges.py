from matplotlib import pyplot as plt
import cv2
import numpy as np


class EdgeDetector:
    @staticmethod
    def sobel_edge_detection(image):
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magnitude *= 255.0 / gradient_magnitude.max()
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        return gradient_magnitude


    @staticmethod
    def roberts_edge_detection(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roberts_kernel_x = np.array([[1, 0],
                                      [0, -1]])
        roberts_kernel_y = np.array([[0, 1],
                                      [-1, 0]])
        gradient_x = cv2.filter2D(gray_image, -1, roberts_kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, roberts_kernel_y)
        gradient_magnitude = np.abs(gradient_x) + np.abs(gradient_y)
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
        return gradient_magnitude.astype(np.uint8)
    
    
    
# test for edge detection 
image_path = "C:/Users/mirna/Downloads/Sobel_src.JPG"
image = cv2.imread(image_path)

if image is not None:
    image_after_sobel = EdgeDetector.sobel_edge_detection(image)
    image_after_robert = EdgeDetector.roberts_edge_detection(image)

    plt.subplot(1, 2, 1)
    plt.imshow(image_after_sobel, cmap='gray')
    plt.title('Sobel Edge Detection')

    plt.subplot(1, 2, 2)
    plt.imshow(image_after_robert, cmap='gray')
    plt.title('Roberts Edge Detection')

    plt.tight_layout()
    plt.show()
else:
    print("Error: Unable to read the image.")
