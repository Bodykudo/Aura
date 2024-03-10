from matplotlib import pyplot as plt
import cv2
import numpy as np


class EdgeDetector:
    @staticmethod
    def sobel_edge_detection(image):
        def sobel_edge_detection(image, direction='both'):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        G = np.zeros(gray_image.shape)
        G_x = np.zeros(gray_image.shape)
        G_y = np.zeros(gray_image.shape)
        size = gray_image.shape
        kernel_x = np.array(([-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]))

        kernel_y = np.array(([-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]))
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                if direction == 'x' or direction == 'both':
                    G_x[i, j] = np.sum(np.multiply(gray_image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
                if direction == 'y' or direction == 'both':
                    G_y[i, j] = np.sum(np.multiply(gray_image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
        if direction == 'x':
            G = np.abs(G_x)
        elif direction == 'y':
            G = np.abs(G_y)
        elif direction == 'both':
            G = np.sqrt(np.square(G_x) + np.square(G_y))
        G = np.multiply(G, 255.0 / G.max())
        G = G.astype('uint8')
        return G


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
    image_after_sobel = EdgeDetector.sobel_edge_detection(image,direction='both')
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
