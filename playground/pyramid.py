import cv2
import numpy as np


def build_scale_space_pyramid(
    image, num_octaves=3, num_scales=5, sigma=1.6, downsampling_factor=2
):
    """
    Constructs a scale-space pyramid representation of a grayscale image.

    A scale-space pyramid is a multi-scale image representation that facilitates
    tasks like feature detection and object recognition. It comprises multiple
    octaves, each containing several scales. Images within a scale are progressively
    blurred versions of the original, capturing features at varying levels of detail.
    Images across octaves are downsampled versions of each other, enabling the
    detection of larger structures.

    Parameters:
        image (numpy.ndarray): The input grayscale image as a NumPy array.
        num_octaves (int, optional): The number of octaves in the pyramid.
            Each octave represents a doubling of the image size. Defaults to 3.
        num_scales (int, optional): The number of scales per octave. Each scale
            represents a level of Gaussian blurring applied to the image. Defaults to 5.
        sigma (float, optional): The initial standard deviation for the
            Gaussian blur kernel, controlling the level of smoothing. Defaults to 1.6.
        downsampling_factor (int, optional): The downsampling factor between
            octaves. Higher values lead to coarser scales but reduce computational
            cost. Defaults to 2 (downsample by half in each octave).

    Returns:
        list: A list of lists representing the scale-space pyramid. The outer list
            contains one sub-list for each octave, and each sub-list contains the
            images for each scale within that octave.
    """

    pyramid = []
    current_image = image.copy()  # Create a copy to avoid modifying the original

    for octave_level in range(num_octaves):
        print(f"Processing octave {octave_level + 1}...")
        octave = []

        for scale_level in range(num_scales):
            print(f"  Building scale {scale_level + 1}...")

            # Apply Gaussian blur for scale invariance
            blurred_image = cv2.GaussianBlur(current_image, (0, 0), sigma)
            octave.append(blurred_image)

            # Increase sigma for the next scale in the octave (effectively scaling blur)
            sigma *= 2.0  # Double sigma for the next scale level

        # Downsample the image for the next octave (reduce resolution)
        new_width = int(current_image.shape[1] / downsampling_factor)
        new_height = int(current_image.shape[0] / downsampling_factor)
        print(f"  Downscaling image to {new_width}x{new_height} for next octave")
        current_image = cv2.resize(
            current_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

        # Reset sigma for the next octave
        sigma = 1.6

        pyramid.append(octave)

    return pyramid


def generate_DoG_pyramid(gaussian_pyramid):
    """
    Generates the Difference-of-Gaussian (DoG) pyramid from the Gaussian pyramid.

    The DoG pyramid highlights image features that are robust across scales, making
    it valuable for tasks like keypoint detection.

    Parameters:
        gaussian_pyramid (list): The Gaussian scale-space pyramid.

    Returns:
        list: A list of lists representing the Difference-of-Gaussian pyramid.
    """

    DoG_pyramid = []
    for octave in gaussian_pyramid:
        DoG_octave = []
        for i in range(len(octave) - 1):
            # Calculate DoG by subtracting consecutive scales within an octave
            DoG_image = octave[i + 1] - octave[i]
            DoG_octave.append(DoG_image)
        DoG_pyramid.append(DoG_octave)

    return DoG_pyramid


# Testing
image_path = r"D:\Raghda\3rd Year\2nd Term\CV\Aura\playground\killua.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

gaussian_pyramid = build_scale_space_pyramid(image)

# Generate the Difference-of-Gaussian (DoG) pyramid
DoG_pyramid = generate_DoG_pyramid(gaussian_pyramid)

# Display all levels of the pyramid (optional)
for octave_level, octave in enumerate(gaussian_pyramid):
    for scale_level, image_level in enumerate(octave):
        print(f"Octave {octave_level + 1}, Scale level {scale_level + 1}:")
        print(f"  Resolution: {image_level.shape[1]}x{image_level.shape[0]}")
        print(f"  Scale: {1.6 * (2 ** scale_level)}\n")
        cv2.imshow(f"Octave {octave_level + 1}, Scale {scale_level + 1}", image_level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Display all levels of the DoG pyramid (optional)
for octave_level, octave in enumerate(DoG_pyramid):
    for scale_level, image_level in enumerate(octave):
        print(f"Octave {octave_level + 1}, Scale level {scale_level + 1}:")
        print(f"  Resolution: {image_level.shape[1]}x{image_level.shape[0]}")
        cv2.imshow(
            f"DoG Octave {octave_level + 1}, Scale {scale_level + 1}", image_level
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("Scale-space pyramid and DoG pyramid generation complete!")
