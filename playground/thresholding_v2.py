import numpy as np
import cv2
import matplotlib.pyplot as plt


def otsu_threshold(image):

    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    pdf = histogram / float(np.sum(histogram))

    cdf = np.cumsum(pdf)

    # Step 4: Compute intensity values
    intensity = np.arange(256)

    mean_intensity = np.sum(intensity * pdf)
    cumulative_sum_intensity = np.cumsum(intensity * pdf)

    omega_0 = cdf
    omega_1 = 1.0 - omega_0
    mean_0 = cumulative_sum_intensity / (omega_0 + 1e-6)
    mean_1 = (mean_intensity - cumulative_sum_intensity) / (omega_1 + 1e-6)
    inter_class_variances = omega_0 * omega_1 * (mean_0 - mean_1) ** 2
    optimal_threshold = np.argmax(inter_class_variances)
    binary_image = (image > optimal_threshold).astype(np.uint8) * 255

    return binary_image, optimal_threshold


def local_otsu_threshold(image, block_size, c):
    height, width = image.shape
    binary_image = np.zeros_like(image)
    # Iterate over blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i : i + block_size, j : j + block_size]
            _, threshold = otsu_threshold(block)
            threshold = threshold - c
            binary_block = (block > threshold).astype(np.uint8) * 255

            binary_image[i : i + block_size, j : j + block_size] = binary_block

    return binary_image


def optimal_threshold(image):
    # Extract intensities of the four corners
    height, width = image.shape
    # Check if the image is empty
    if height == 0 or width == 0:
        # Handle the case of an empty image
        # You can return a default threshold and a blank binary image, or raise an error
        return np.zeros_like(image, dtype=np.uint8), 0
    corners = [
        image[0, 0],
        image[0, width - 1],
        image[height - 1, 0],
        image[height - 1, width - 1],
    ]

    threshold = np.mean(corners)

    while True:
        class1_mean = np.mean(image[image < threshold])
        class2_mean = np.mean(image[image >= threshold])

        new_threshold = (class1_mean + class2_mean) / 2

        # Check for convergence
        if np.abs(new_threshold - threshold) < 1e-6:
            break

        threshold = new_threshold

    binary_image = (image > threshold).astype(np.uint8) * 255
    
    
    return binary_image, threshold


def adaptive_local_threshold(image, block_size, c):
    height, width = image.shape
    binary_image = np.zeros_like(image)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Define the region of interest (ROI) for the current block
            roi = image[i : i + block_size, j : j + block_size]
            _, threshold = optimal_threshold(roi)

            threshold = threshold - c
            binary_roi = (roi > threshold).astype(np.uint8) * 255
            binary_image[i : i + block_size, j : j + block_size] = binary_roi

    return binary_image


# Example usage:
# Assuming 'image' is your grayscale input image
image = cv2.imread("./playground/cameraman.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding using the custom implementation
binary_image_otsu_custom, th = otsu_threshold(image)
print(f"OTSU: {th}")

# Apply optimal thresholding using the custom implementation
binary_image_optimal_custom, th2 = optimal_threshold(image)
print(f"Optimal: {th2}")
# Apply Otsu's thresholding using OpenCV's built-in method for comparison
th3, binary_image_otsu_opencv = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"OPENCV: {th3}")
# Define the block size for local Otsu thresholding
block_size = 11  # Adjust this value as needed
c = 7
# Apply local Otsu's thresholding using the custom implementation
binary_image_local_otsu_custom = local_otsu_threshold(image, block_size, c)

# Apply local Otsu's thresholding using the optimal implementation
binary_image_local_optimal_custom = adaptive_local_threshold(image, block_size, c)

# Apply local Otsu's thresholding using OpenCV's built-in method
binary_image_local_opencv = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
)

# Plotting all binary images in a single figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(binary_image_otsu_custom, cmap="gray")
plt.title("Custom Otsu Thresholding")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(binary_image_optimal_custom, cmap="gray")
plt.title("Custom Optimal Thresholding")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(binary_image_otsu_opencv, cmap="gray")
plt.title("OpenCV Otsu Thresholding")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(binary_image_local_otsu_custom, cmap="gray")
plt.title("Local Otsu Thresholding (Custom)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(binary_image_local_optimal_custom, cmap="gray")
plt.title("Local Optimal Thresholding (Custom)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(binary_image_local_opencv, cmap="gray")
plt.title("adaptive Thresholding (OpenCV)")
plt.axis("off")

plt.tight_layout()
plt.show()
