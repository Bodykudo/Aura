import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class Histogram:
    """
    A class for working with image histograms.

    Provides methods for calculating, plotting, equalizing, and normalizing histograms for both grayscale and RGB images.
    """

    @staticmethod
    def convert_to_grayscale(image):
        # Convert image to grayscale using luminosity method
        gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        # Convert back to uint8 format for display
        gray_image = gray_image.astype(np.uint8)
        return gray_image

    @staticmethod
    def calculate_histogram_gray_scale(image, min_range=0, max_range=256):

        # image = Histogram.convert_to_grayscale(image)

        hist, bins = np.histogram(
            image.flatten(), bins=256, range=[min_range, max_range]
        )

        return hist

    @staticmethod
    def calculate_histogram_RGB(image, min_range=0, max_range=256):

        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        # Calculate histograms for each color channel
        blue_hist = np.histogram(
            blue_channel.flatten(), bins=256, range=(min_range, max_range)
        )
        green_hist = np.histogram(
            green_channel.flatten(), bins=256, range=(min_range, max_range)
        )
        red_hist = np.histogram(
            red_channel.flatten(), bins=256, range=(min_range, max_range)
        )

        return blue_hist, green_hist, red_hist

    @staticmethod
    def calcualte_histogram(image, min_range=0, max_range=255):
        if len(image.shape) == 2:
            final_hist = Histogram.calculate_histogram_gray_scale(
                image, min_range, max_range
            )

        elif len(image.shape) == 3:
            final_hist = Histogram.calculate_histogram_RGB(image, min_range, max_range)

        return final_hist

    @staticmethod
    def calculate_cumulative_distribution_gray_scale(image):

        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())

        return cdf_normalized

    @staticmethod
    def calculate_cumulative_distribution_RGB(image):

        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]

        # Calculate histograms for each color channel
        hist_b, _ = np.histogram(b.flatten(), bins=256, range=(0, 256))
        hist_g, _ = np.histogram(g.flatten(), bins=256, range=(0, 256))
        hist_r, _ = np.histogram(r.flatten(), bins=256, range=(0, 256))

        # Calculate cumulative distribution function (CDF) for each channel
        cdf_b = hist_b.cumsum()
        cdf_g = hist_g.cumsum()
        cdf_r = hist_r.cumsum()

        cdf_b = cdf_b / float(cdf_b.max())
        cdf_g = cdf_g / float(cdf_g.max())
        cdf_r = cdf_r / float(cdf_r.max())

        return cdf_b, cdf_g, cdf_r

    @staticmethod
    def calculate_cumulative_distribution(image):

        if len(image.shape) == 2:
            final_cdf = Histogram.calculate_cumulative_distribution_gray_scale(image)

        elif len(image.shape) == 3:
            final_cdf = Histogram.calculate_cumulative_distribution_RGB(image)

        return final_cdf

    @staticmethod
    def plot_histogram_gray_scale(image):

        plt.hist(image.ravel(), bins=256, range=[0, 1], color="black")

        plt.xlabel("Intensity Value")
        plt.ylabel("Pixel Count")
        plt.title("Histogram of the Grayscale Image")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_histogram_RGB(image):

        b, g, r = cv2.split(image)
        plt.hist(r.ravel(), bins=256, range=[0.0, 255.0], color="r")
        plt.hist(g.ravel(), bins=256, range=[0.0, 255.0], color="g")
        plt.hist(b.ravel(), bins=256, range=[0.0, 255.0], color="b")

        plt.xlabel("Intensity Value")
        plt.ylabel("Pixel Count")
        plt.title("Histogram of the Grayscale Image")
        plt.grid(True)
        plt.show()

    @staticmethod
    def equalize_histogram_gray_scale(image):
        """
        Performs histogram equalization on a grayscale image.

        Args:
            image: A grayscale image represented as a 2D NumPy array.

        Returns:
            A new grayscale image with equalized histogram.
        """
        # image = Histogram.convert_to_grayscale(image)
        # Get image histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

        # Calculate cumulative distribution function (CDF)
        cdf = hist.copy()
        cdf = np.cumsum(cdf)

        # Normalize CDF
        cdf = cdf / cdf[-1]
        cdf = np.round(cdf * 255).astype(np.uint8)  # Map to intensity range

        # Apply lookup table (CDF) for equalization
        img_eq = cdf[image]

        return img_eq

    @staticmethod
    def equalize_histogram_RGB(image):
        b, g, r = cv2.split(image)

        hist_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        hist_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        hist_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])

        cdf_b = np.cumsum(hist_b)
        cdf_g = np.cumsum(hist_g)
        cdf_r = np.cumsum(hist_r)

        cdf_m_b = np.ma.masked_equal(cdf_b, 0)
        cdf_m_b = (cdf_m_b) * 255 / (cdf_m_b.max())
        cdf_final_b = np.ma.filled(cdf_m_b, 0).astype("uint8")

        cdf_m_g = np.ma.masked_equal(cdf_g, 0)
        cdf_m_g = (cdf_m_g) * 255 / (cdf_m_g.max())
        cdf_final_g = np.ma.filled(cdf_m_g, 0).astype("uint8")

        cdf_m_r = np.ma.masked_equal(cdf_r, 0)
        cdf_m_r = (cdf_m_r) * 255 / (cdf_m_r.max())
        cdf_final_r = np.ma.filled(cdf_m_r, 0).astype("uint8")

        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))
        return img_out

    @staticmethod
    def equalize_histogram(image):

        if len(image.shape) == 2:
            image = Histogram.equalize_histogram_gray_scale(image)

        elif len(image.shape) == 3:
            image = Histogram.equalize_histogram_RGB(image)

        return image

    @staticmethod
    def normalize_image_gray_scale(image):
        """Normalizes an image to the range [0, 1]."""
        image = image.astype(np.float32)
        maxVal = np.max(image)
        minVal = np.min(image)
        image = (image - minVal) / (maxVal - minVal)
        return image

    @staticmethod
    def normalize_image_RGB(image):

        image = image.astype(np.float32)

        for i in range(3):  # Loop through each color channel (R, G, B)
            min_val = np.min(image[:, :, i])
            max_val = np.max(image[:, :, i])
            image[:, :, i] = (image[:, :, i] - min_val) / (max_val - min_val)

        return image

    @staticmethod
    def normalize_image(image):

        if len(image.shape) == 2:
            image = Histogram.normalize_image_gray_scale(image)

        elif len(image.shape) == 3:
            image = Histogram.normalize_image_RGB(image)

        return image


# if __name__ == "__main__":
# ------------------------------------------------READ IMAGE-----------------------------------------------------------#
# image = cv2.imread(".\playground\image1.jpg")

# Histogram = Histogram()

# ------------------------------------------------------TEST Histogram--------------------------------------------------#
# image_histogram = Histogram.calculate_histogram_gray_scale(image)
# blue_hist, green_hist, red_hist = Histogram.calculate_histogram_RGB(image)

# # Plot the histograms
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 4, 1)
# plt.plot(image_histogram,  color='black')
# plt.title('Image Histogram')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.subplot(1, 4, 2)
# plt.plot(blue_hist[1][:-1], blue_hist[0], color='blue')
# plt.title('Blue Channel Histogram')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.subplot(1, 4, 3)
# plt.plot(green_hist[1][:-1], green_hist[0], color='green')
# plt.title('Green Channel Histogram')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.subplot(1, 4, 4)
# plt.plot(red_hist[1][:-1], red_hist[0], color='red')
# plt.title('Red Channel Histogram')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.show()

# ------------------------------------------------------ TEST Equalization --------------------------------------------------#

# equalized_image = Histogram.equalize_histogram_gray_scale(image)

# equalized_image_cv2 = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap="gray")  # Convert BGR to RGB for displaying with matplotlib
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(equalized_image, cmap="gray")  # Assuming grayscale image
# plt.title("equalized image mathmatically")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(equalized_image_cv2, cmap="gray")  # Assuming grayscale image
# plt.title("equalized image cv2")
# plt.axis("off")

# plt.show()
# ------------------------------------------------------TEST NORMALIZATION--------------------------------------------------#

# image_normalized = Histogram.normalize_image(image)
# image = image.astype(np.float32)
# image_normalized_cv2 = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# plt.figure(figsize=(10, 5))

# plt.subplot(2, 3, 1)
# plt.imshow(image,  cmap='gray')  # Convert BGR to RGB for displaying with matplotlib
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.imshow(image_normalized, cmap='gray')  # Assuming grayscale image
# plt.title('Normalized Image mathmatically')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.imshow(image_normalized_cv2, cmap='gray')  # Assuming grayscale image
# plt.title('Normalized Image with cv2')
# plt.axis('off')

# plt.subplot(2, 3, 4)
# plt.hist(image.ravel(), 256, [0,256], color ='gray') # Assuming grayscale image
# plt.title('histogram_for_original_image')

# plt.subplot(2, 3, 5)
# plt.hist(image_normalized.flatten(), bins=256, range=[0, 1], color='black')# Assuming grayscale image
# plt.title('Histogram of Normalized Image')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.subplot(2, 3, 6)
# plt.hist(image_normalized_cv2.flatten(), bins=256, range=[0, 1], color='black')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')# Assuming grayscale image
# plt.title('histogram_for_normalized_image_cv2')

# plt.show()
