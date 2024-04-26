import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(image):
    if len(image.shape) > 2:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image

    histogram = np.histogram(grayscale_image, bins=256, range=[0, 256])[0]
    cumulative_histogram = np.cumsum(histogram)
    global_mean_intensity = np.dot(np.arange(256), histogram) / histogram.sum()
    return grayscale_image, histogram, cumulative_histogram, global_mean_intensity


def find_spectral_thresholds(histogram, global_mean_intensity):
    max_variance = 0
    for high_threshold in range(1, 256):
        for low_threshold in range(high_threshold):
            weights = np.array(
                [
                    histogram[:low_threshold].sum(),
                    histogram[low_threshold:high_threshold].sum(),
                    histogram[high_threshold:].sum(),
                ]
            )
            if weights[1] == 0:  # Skip if weight is 0 to avoid division by zero
                continue
            means = np.array(
                [
                    np.dot(np.arange(0, low_threshold), histogram[:low_threshold])
                    / weights[0],
                    np.dot(
                        np.arange(low_threshold, high_threshold),
                        histogram[low_threshold:high_threshold],
                    )
                    / weights[1],
                    np.dot(np.arange(high_threshold, 256), histogram[high_threshold:])
                    / weights[2],
                ]
            )
            variance = np.dot(weights, (means - global_mean_intensity) ** 2)
            if variance > max_variance:
                max_variance = variance
                optimal_low_threshold, optimal_high_threshold = (
                    low_threshold,
                    high_threshold,
                )

    return optimal_low_threshold, optimal_high_threshold


def apply_threshold(image, low_threshold, high_threshold):
    binary_image = np.zeros_like(image)
    binary_image[image < low_threshold] = 0
    binary_image[(image >= low_threshold) & (image < high_threshold)] = 128
    binary_image[image >= high_threshold] = 255
    return binary_image


def spectral_thresholding(image):
    grayscale_image, histogram, _, global_mean_intensity = preprocess(image)
    optimal_low_threshold, optimal_high_threshold = find_spectral_thresholds(
        histogram, global_mean_intensity
    )
    binary_image = apply_threshold(
        grayscale_image, optimal_low_threshold, optimal_high_threshold
    )
    return binary_image


def spectral_thresholding_local(image, size):
    result = np.zeros(
        image.shape[:2], dtype=np.uint8
    )  # Initialize as a grayscale image
    image_height, image_width = image.shape[:2]
    step_y, step_x = size

    for y in range(0, image_height, step_y):
        for x in range(0, image_width, step_x):
            sub_image = image[
                y : min(y + step_y, image_height), x : min(x + step_x, image_width)
            ]
            local_thresholded = spectral_thresholding(sub_image)

            result[
                y : min(y + step_y, image_height), x : min(x + step_x, image_width)
            ] = local_thresholded

    return result


# ---------------------TESTING---------------------------------
# Load the image
input_image = cv2.imread(
    r"d:\Raghda\3rd Year\2nd Term\CV\Aura\playground\cameraman.jpg", cv2.IMREAD_COLOR
)

# Convert the original image to grayscale
input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply spectral thresholding
thresholded_image = spectral_thresholding(input_image)

# Convert the thresholded image to RGB for display
thresholded_image_rgb = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)

# Apply local spectral thresholding
window_size = (100, 100)
local_thresholded_image = spectral_thresholding_local(input_image, window_size)

# Display the original, global thresholded, and local thresholded images
plt.figure(figsize=(15, 5))

# Original Image (Black and White)
plt.subplot(1, 3, 1)
plt.imshow(input_image_gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Global Thresholded Image
plt.subplot(1, 3, 2)
plt.imshow(thresholded_image_rgb)
plt.title("Global Thresholded Image")
plt.axis("off")

# Local Thresholded Image
plt.subplot(1, 3, 3)
plt.imshow(local_thresholded_image, cmap="gray")
plt.title("Local Thresholded Image")
plt.axis("off")
plt.show()


# --------------------OLD THRESHOLDING-------------------------
# import numpy as np

# from api.utils import read_image


# class Thresholding:
#     @staticmethod
#     def global_thresholding(image_path: str, threshold: int = 127):
#         image = read_image(image_path, grayscale=True)
#         return np.where(image > threshold, 255, 0).astype(np.uint8)

#     @staticmethod
#     def local_thresholding(
#         image_path: str,
#         threshold_margin: int,
#         block_size: int,
#     ):
#         window_size = (block_size, block_size)
#         image = read_image(image_path, grayscale=True)
#         height, width = image.shape
#         output = np.zeros_like(image)
#         for i in range(height):
#             for j in range(width):
#                 window = image[
#                     max(0, i - window_size[0] // 2) : min(
#                         height, i + window_size[0] // 2 + 1
#                     ),
#                     max(0, j - window_size[1] // 2) : min(
#                         width, j + window_size[1] // 2 + 1
#                     ),
#                 ]
#                 local_mean = np.mean(window.flatten())
#                 output[i, j] = 255 if image[i, j] > local_mean - threshold_margin else 0
#         return output
