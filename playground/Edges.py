from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.exposure as exposure
from scipy.ndimage import convolve


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif 22.5 <= angles[i, j] < 67.5:
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif 67.5 <= angles[i, j] < 112.5:
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]

    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
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
            if (new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (
                result[new_x, new_y] == weak
            ):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)
        ),
        (size, size),
    )
    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel_size, sigma):
    kernel_size = int(kernel_size)
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = convolve(image, kernel)
    return blurred_image


class EdgeDetector:
    @staticmethod
    def sobel_edge_detection(image, gaussian_ksize, sigma, direction="both"):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gaussian_blur(gray_image, gaussian_ksize, 1.0)
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobelx = cv2.filter2D(gray_image, cv2.CV_32F, sobel_x_kernel)
        sobely = cv2.filter2D(gray_image, cv2.CV_32F, sobel_y_kernel)
        sobelx_norm = (
            exposure.rescale_intensity(sobelx, in_range="image", out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        sobely_norm = (
            exposure.rescale_intensity(sobely, in_range="image", out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        sobel_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel_magnitude = (
            exposure.rescale_intensity(
                sobel_magnitude, in_range="image", out_range=(0, 255)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        phase = np.rad2deg(np.arctan2(sobely, sobelx))
        phase[phase < 0] += 180

        if direction == "x":
            return sobelx_norm, phase
        elif direction == "y":
            return sobely_norm, phase
        else:
            return sobel_magnitude, phase

    @staticmethod
    def prewitt_edge_detection(image, gaussian_ksize):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (gaussian_ksize, gaussian_ksize), 0)
        prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

        prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(gray_image, cv2.CV_32F, prewitt_kernel_x)
        prewitty = cv2.filter2D(gray_image, cv2.CV_32F, prewitt_kernel_y)
        magnitude = np.sqrt(np.square(prewittx) + np.square(prewitty))
        prewitt_magnitude = (
            exposure.rescale_intensity(magnitude, in_range="image", out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        return prewitt_magnitude

    @staticmethod
    def roberts_edge_detection(image, gaussian_ksize):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (gaussian_ksize, gaussian_ksize), 0)
        roberts_kernel_x = np.array([[1, 0], [0, -1]])
        roberts_kernel_y = np.array([[0, 1], [-1, 0]])
        gradient_x = cv2.filter2D(gray_image, -1, roberts_kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, roberts_kernel_y)
        gradient_magnitude = np.abs(gradient_x) + np.abs(gradient_y)
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
        return gradient_magnitude.astype(np.uint8)

    @staticmethod
    def canny_edge_detection(
        image, gaussian_ksize, sigma, low_threshold, high_threshold
    ):
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurred_image = cv2.GaussianBlur(
        # gray_image, (gaussian_ksize, gaussian_ksize), sigma
        # )
        # edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
        # return edges
        image, angles = EdgeDetector.sobel_edge_detection(
            image, gaussian_ksize, sigma, "both"
        )
        image = non_maximum_suppression(image, angles)
        image = double_threshold_hysteresis(image, low_threshold, high_threshold)
        return image


# test for edge detection
image_path = "image1.jpg"
image = cv2.imread(image_path)

if image is not None:
    image_after_sobel, _ = EdgeDetector.sobel_edge_detection(image, 3, 1, direction="x")
    image_after_robert = EdgeDetector.roberts_edge_detection(image, 3)
    image_after_canny = EdgeDetector.canny_edge_detection(
        image, gaussian_ksize=5, sigma=0, low_threshold=50, high_threshold=150
    )
    image_after_prewitt = EdgeDetector.prewitt_edge_detection(image, 3)

    plt.subplot(2, 2, 1)
    plt.imshow(image_after_prewitt, cmap="gray")
    plt.title("image_after_prewitt")

    plt.subplot(2, 2, 2)
    plt.imshow(image_after_canny, cmap="gray")
    plt.title("image_after_canny")

    plt.subplot(2, 2, 3)
    plt.imshow(image_after_sobel, cmap="gray")
    plt.title("image_after_sobel")

    plt.subplot(2, 2, 4)
    plt.imshow(image_after_robert, cmap="gray")
    plt.title("image_after_robert")

    plt.tight_layout()
    plt.show()
else:
    print("Error: Unable to read the image.")
