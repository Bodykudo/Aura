from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.exposure as exposure



class EdgeDetector:
    @staticmethod
    def sobel_edge_detection(image, gaussian_ksize, direction='both'):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (gaussian_ksize, gaussian_ksize), 0)
        roberts_kernel_x = np.array([[1, 0],
                                    [0, -1]])
        roberts_kernel_y = np.array([[0, 1],
                                    [-1, 0]])
        sobelx = cv2.filter2D(gray_image, cv2.CV_32F, roberts_kernel_x)
        sobely = cv2.filter2D(gray_image, cv2.CV_32F, roberts_kernel_y)
        sobelx_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
        sobely_norm = exposure.rescale_intensity(sobely, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
        sobel_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
        
        if direction == 'x':
            return sobelx_norm
        elif direction == 'y':
            return sobely_norm
        else:
            return sobel_magnitude
        

        
    @staticmethod
    def prewitt_edge_detection(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        prewitt_kernel_x = np.array([[1, 0, -1],
                                     [1, 0, -1],
                                     [1, 0, -1]])

        prewitt_kernel_y = np.array([[1, 1, 1],
                                     [0, 0, 0],
                                     [-1, -1, -1]])
        prewittx = cv2.filter2D(gray_image, cv2.CV_32F, prewitt_kernel_x)
        prewitty = cv2.filter2D(gray_image, cv2.CV_32F, prewitt_kernel_y)
        magnitude = np.sqrt(np.square(prewittx) + np.square(prewitty))
        prewitt_magnitude = exposure.rescale_intensity(magnitude, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
        return prewitt_magnitude
        


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
    

    @staticmethod
    def canny_edge_detection(image, gaussian_ksize, low_threshold, high_threshold):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (gaussian_ksize, gaussian_ksize), 0)
        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
        return edges
    
    
    
# test for edge detection 
image_path = "C:/Users/mirna/Downloads/test.PNG"
image = cv2.imread(image_path)

if image is not None:
    image_after_sobel = EdgeDetector.sobel_edge_detection(image,3,direction='both')
    image_after_robert = EdgeDetector.roberts_edge_detection(image)
    image_after_canny=EdgeDetector.canny_edge_detection(image, gaussian_ksize=5, low_threshold=10, high_threshold=50)
    image_after_prewitt=EdgeDetector.prewitt_edge_detection(image)

    plt.subplot(1, 2, 1)
    plt.imshow(image_after_prewitt, cmap='gray')
    plt.title('image_after_prewitt')

    plt.subplot(1, 2, 2)
    plt.imshow(image_after_sobel, cmap='gray')
    plt.title('image_after_sobel')


    plt.tight_layout()
    plt.show()
else:
    print("Error: Unable to read the image.")
