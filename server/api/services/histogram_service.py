import numpy as np
import cv2


class Histogram:
    @staticmethod
    def grayscale_image(image):

        gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        gray_image = gray_image.astype(np.uint8)
        return gray_image

    @staticmethod
    def get_grayscale_histogram(image, min_range=0, max_range=256):
        histogram, _ = np.histogram(
            image.flatten(), bins=256, range=[min_range, max_range]
        )
        return histogram

    @staticmethod
    def get_rgb_histogram(image, min_range=0, max_range=256):
        channel_blue = image[:, :, 0]
        channel_green = image[:, :, 1]
        channel_red = image[:, :, 2]

        histogram_blue = np.histogram(
            channel_blue.flatten(), bins=256, range=(min_range, max_range)
        )
        histogram_green = np.histogram(
            channel_green.flatten(), bins=256, range=(min_range, max_range)
        )
        histogram_red = np.histogram(
            channel_red.flatten(), bins=256, range=(min_range, max_range)
        )

        return histogram_blue, histogram_green, histogram_red

    @staticmethod
    def get_histogram(image, min_range=0, max_range=256):
        if len(image.shape) == 2:
            histogram = Histogram.get_grayscale_histogram(image, min_range, max_range)
            histogram_data = [
                {"name": int(i), "gray": int(histogram[i])}
                for i in range(len(histogram))
            ]
        elif len(image.shape) == 3:
            histogram = Histogram.get_rgb_histogram(image, min_range, max_range)
            red_histogram = histogram[2][0]
            green_histogram = histogram[1][0]
            blue_histogram = histogram[0][0]
            if np.array_equal(red_histogram, green_histogram) and np.array_equal(
                green_histogram, blue_histogram
            ):
                histogram_data = [
                    {"name": int(i), "gray": int(red_histogram[i])}
                    for i in range(len(red_histogram))
                ]
            else:
                histogram_data = [
                    {
                        "name": int(i),
                        "red": int(red_histogram[i]),
                        "green": int(green_histogram[i]),
                        "blue": int(blue_histogram[i]),
                    }
                    for i in range(len(red_histogram))
                ]

        return histogram_data

    @staticmethod
    def get_grayscale_cdf(image, min_range=0, max_range=256):
        hist, _ = np.histogram(image.flatten(), bins=256, range=[min_range, max_range])
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())

        return cdf_normalized

    @staticmethod
    def get_rgb_cdf(image, min_range=0, max_range=256):
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]

        # Calculate histograms for each color channel
        histogram_blue, _ = np.histogram(
            blue.flatten(), bins=256, range=(min_range, max_range)
        )
        histogram_green, _ = np.histogram(
            green.flatten(), bins=256, range=(min_range, max_range)
        )
        histogram_red, _ = np.histogram(
            red.flatten(), bins=256, range=(min_range, max_range)
        )

        # Calculate cumulative distribution function (CDF) for each channel
        cdf_blue = histogram_blue.cumsum()
        cdf_green = histogram_green.cumsum()
        cdf_red = histogram_red.cumsum()

        cdf_blue = cdf_blue / float(cdf_blue.max())
        cdf_green = cdf_green / float(cdf_green.max())
        cdf_red = cdf_red / float(cdf_red.max())

        return cdf_blue, cdf_green, cdf_red

    @staticmethod
    def get_cdf(image, min_range=0, max_range=256):
        if len(image.shape) == 2:
            cdf = Histogram.get_grayscale_cdf(image, min_range, max_range)
            cdf_data = [
                {"name": int(i), "gray": float(cdf[i])} for i in range(len(cdf))
            ]
        elif len(image.shape) == 3:
            cdf = Histogram.get_rgb_cdf(image, min_range, max_range)
            red_cdf = cdf[2]
            green_cdf = cdf[1]
            blue_cdf = cdf[0]
            if np.array_equal(red_cdf, green_cdf) and np.array_equal(
                green_cdf, blue_cdf
            ):
                cdf_data = [
                    {"name": int(i), "gray": float(red_cdf[i])}
                    for i in range(len(red_cdf))
                ]
            else:
                cdf_data = [
                    {
                        "name": int(i),
                        "red": float(red_cdf[i]),
                        "green": float(green_cdf[i]),
                        "blue": float(blue_cdf[i]),
                    }
                    for i in range(len(red_cdf))
                ]
        return cdf_data

    @staticmethod
    def equalize_grayscale(image):
        """
        Performs histogram equalization on a grayscale image.

        Args:
            image: A grayscale image represented as a 2D NumPy array.

        Returns:
            A new grayscale image with equalized histogram.
        """
        # Get image histogram
        histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])

        # Calculate cumulative distribution function (CDF)
        cdf = histogram.copy()
        cdf = np.cumsum(cdf)

        # Normalize CDF
        cdf = cdf / cdf[-1]
        cdf = np.round(cdf * 255).astype(np.uint8)  # Map to intensity range

        # Apply lookup table (CDF) for equalization
        equalized_image = cdf[image]

        return equalized_image

    @staticmethod
    def equalize_rgb(image):
        blue, green, red = cv2.split(image)

        histogram_blue, _ = np.histogram(blue.flatten(), 256, [0, 256])
        histogram_green, _ = np.histogram(green.flatten(), 256, [0, 256])
        histogram_red, _ = np.histogram(red.flatten(), 256, [0, 256])

        cdf_blue = np.cumsum(histogram_blue)
        cdf_green = np.cumsum(histogram_green)
        cdf_red = np.cumsum(histogram_red)

        masked_cdf_blue = np.ma.masked_equal(cdf_blue, 0)
        masked_cdf_blue = (masked_cdf_blue) * 255 / (masked_cdf_blue.max())
        final_cdf_blue = np.ma.filled(masked_cdf_blue, 0).astype("uint8")

        masked_cdf_green = np.ma.masked_equal(cdf_green, 0)
        masked_cdf_green = (masked_cdf_green) * 255 / (masked_cdf_green.max())
        final_cdf_green = np.ma.filled(masked_cdf_green, 0).astype("uint8")

        masked_cdf_red = np.ma.masked_equal(cdf_red, 0)
        masked_cdf_red = (masked_cdf_red) * 255 / (masked_cdf_red.max())
        final_cdf_red = np.ma.filled(masked_cdf_red, 0).astype("uint8")

        image_blue = final_cdf_blue[blue]
        image_green = final_cdf_green[green]
        image_red = final_cdf_red[red]

        equalized_image = cv2.merge((image_blue, image_green, image_red))
        return equalized_image

    @staticmethod
    def equalize_image(image):
        if len(image.shape) == 2:
            image = Histogram.equalize_grayscale(image)
        elif len(image.shape) == 3:
            image = Histogram.equalize_rgb(image)
        return image

    @staticmethod
    def normalize_grayscale(image):
        """Normalizes an image to the range [0, 1]."""
        normalized_image = image.astype(np.float32)
        min_value = np.min(normalized_image)
        max_value = np.max(normalized_image)
        normalized_image = (normalized_image - min_value) / (max_value - min_value)
        return normalized_image

    @staticmethod
    def normalize_rgb(image):
        normalized_image = image.astype(np.float32)
        for i in range(3):
            min_value = np.min(normalized_image[:, :, i])
            max_value = np.max(normalized_image[:, :, i])
            normalized_image[:, :, i] = (normalized_image[:, :, i] - min_value) / (
                max_value - min_value
            )
        return normalized_image

    @staticmethod
    def normalize_image(image):
        if len(image.shape) == 2:
            image = Histogram.normalize_grayscale(image)
        elif len(image.shape) == 3:
            image = Histogram.normalize_rgb(image)
        return image
