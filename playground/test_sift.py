
import time
from numpy import (
    all,
    any,
    array,
    arctan2,
    cos,
    sin,
    exp,
    dot,
    log,
    logical_and,
    roll,
    sqrt,
    stack,
    trace,
    unravel_index,
    pi,
    deg2rad,
    rad2deg,
    where,
    zeros,
    floor,
    full,
    nan,
    isnan,
    round,
    float32,
)
from numpy.linalg import det, lstsq, norm
from cv2 import (
    drawKeypoints,
    imread,
    imshow,
    resize,
    GaussianBlur,
    subtract,
    KeyPoint,
    INTER_LINEAR,
    INTER_NEAREST,
    waitKey,
    imread,
)
import cv2
from functools import cmp_to_key

import numpy as np

#############################################################################################################
float_tolerance = 1e-7


#############################################################################################################
############################################   MIRNA #################################################################
def SIFT(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image"""
    image = image.astype("float32")
    base_image = createModifiedImage(image, sigma, assumed_blur)
    num_octaves = calculateOctaveCount(base_image.shape)
    gaussian_kernels = getGaussianFilterSizes(sigma, num_intervals)
    gaussian_images = getPyramid(base_image, num_octaves, gaussian_kernels)
    dog_images = getDOG(gaussian_images)
    keypoints = detectScaleSpaceExtrema(
        gaussian_images, dog_images, num_intervals, sigma, image_border_width
    )
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors


############################################   MIRNA #################################################################
def createModifiedImage(original_image, sigma_value, blur_estimate):
    # Increase the size of the original image and calculate the new sigma
    modified_image = resize(
        original_image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR
    )
    adjusted_sigma = sqrt(max((sigma_value**2) - ((2 * blur_estimate) ** 2), 0.01))
    return GaussianBlur(
        modified_image, (0, 0), sigmaX=adjusted_sigma, sigmaY=adjusted_sigma
    )  # the image blur is now sigma instead of assumed_blur


#############################################################################################################
def calculateOctaveCount(image_dimensions):
    """Determine the number of octaves in the image pyramid."""
    return int(round(log(min(image_dimensions)) / log(2) - 1))


#############################################################################################################
def getGaussianFilterSizes(sigma_value, interval_count):
    """Produce a sequence of Gaussian filter sizes for blurring the input image.
    Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper."""
    images_per_octave = interval_count + 3
    k_value = 2 ** (1.0 / interval_count)
    gaussian_filter_sizes = zeros(
        images_per_octave
    )  # scale of gaussian blur needed to transition from one blur scale to the next within an octave
    gaussian_filter_sizes[0] = sigma_value

    for index in range(1, images_per_octave):
        previous_sigma = (k_value ** (index - 1)) * sigma_value
        total_sigma = k_value * previous_sigma
        gaussian_filter_sizes[index] = sqrt(total_sigma**2 - previous_sigma**2)
    return gaussian_filter_sizes


#############################################################################################################
def getPyramid(image_data, octave_count, gaussian_filters):
    """Create a Gaussian scale-space pyramid of images."""
    gaussian_pyramid = []

    for octave_index in range(octave_count):
        octave_images = []
        octave_images.append(
            image_data
        )  # The first image in an octave already has the correct blur
        for filter_size in gaussian_filters[1:]:
            image_data = cv2.GaussianBlur(
                image_data, (0, 0), sigmaX=filter_size, sigmaY=filter_size
            )
            octave_images.append(image_data)
        gaussian_pyramid.append(octave_images)
        base_image = octave_images[-3]
        image_data = resize(
            base_image,
            (int(base_image.shape[1] / 2), int(base_image.shape[0] / 2)),
            interpolation=INTER_NEAREST,
        )
    return array(gaussian_pyramid, dtype=object)


#############################################################################################################
def getDOG(pyramid):
    """Create a pyramid of Difference-of-Gaussians images."""
    dog_pyramid = []

    for octave_images in pyramid:
        dog_images_in_octave = []
        for first_img, second_img in zip(octave_images, octave_images[1:]):
            dog_images_in_octave.append(
                subtract(second_img, first_img)
            )  # Regular subtraction won't suffice as the images are unsigned integers
        dog_pyramid.append(dog_images_in_octave)
    return array(dog_pyramid, dtype=object)


#############################################################################################################
def detectScaleSpaceExtrema(
    gaussian_pyramid,
    dog_pyramid,
    interval_count,
    sigma_value,
    border_width,
    contrast_threshold=0.04,
):
    """Identify the positions of all scale-space extrema in the image pyramid."""
    threshold_value = floor(
        0.5 * contrast_threshold / interval_count * 255
    )  # Taken from OpenCV's implementation
    found_keypoints = []

    for octave_index, dog_octave in enumerate(dog_pyramid):
        for image_index, (first_img, second_img, third_img) in enumerate(
            zip(dog_octave, dog_octave[1:], dog_octave[2:])
        ):
            # (x, y) represents the center of the 3x3 array
            for x in range(border_width, first_img.shape[0] - border_width):
                for y in range(border_width, first_img.shape[1] - border_width):
                    if check_pixel_extremum(
                        first_img[x - 1 : x + 2, y - 1 : y + 2],
                        second_img[x - 1 : x + 2, y - 1 : y + 2],
                        third_img[x - 1 : x + 2, y - 1 : y + 2],
                        threshold_value,
                    ):
                        extremum_result = localize_extremum_via_quadratic_fit(
                            x,
                            y,
                            image_index + 1,
                            octave_index,
                            interval_count,
                            dog_octave,
                            sigma_value,
                            contrast_threshold,
                            border_width,
                        )
                        if extremum_result is not None:
                            keypoint, localized_image_index = extremum_result
                            keypoints_with_orientations = (
                                computeKeypointsWithOrientations(
                                    keypoint,
                                    octave_index,
                                    gaussian_pyramid[octave_index][
                                        localized_image_index
                                    ],
                                )
                            )
                            for (
                                keypoint_with_orientation
                            ) in keypoints_with_orientations:
                                found_keypoints.append(keypoint_with_orientation)
    return found_keypoints


###########################    HABIBA #########################
#############################################################################################################
def check_pixel_extremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise"""

    center_pixel_value = second_subimage[1, 1]

    if abs(center_pixel_value) <= threshold:
        return False

    if center_pixel_value > 0:
        return (
            all(center_pixel_value >= first_subimage)
            and all(center_pixel_value >= third_subimage)
            and all(center_pixel_value >= second_subimage[0, :])
            and all(center_pixel_value >= second_subimage[2, :])
            and center_pixel_value >= second_subimage[1, 0]
            and center_pixel_value >= second_subimage[1, 2]
        )
    elif center_pixel_value < 0:
        return (
            all(center_pixel_value <= first_subimage)
            and all(center_pixel_value <= third_subimage)
            and all(center_pixel_value <= second_subimage[0, :])
            and all(center_pixel_value <= second_subimage[2, :])
            and center_pixel_value <= second_subimage[1, 0]
            and center_pixel_value <= second_subimage[1, 2]
        )


#############################################################################################################
def localize_extremum_via_quadratic_fit(
    row_index,
    col_index,
    image_index,
    octave_index,
    num_intervals,
    dog_images_in_octave,
    sigma,
    contrast_threshold,
    image_border_width,
    eigenvalue_ratio=10,
    num_attempts_until_convergence=5,
):
    """Iteratively refines pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors"""

    is_extremum_outside_image = False
    image_shape = dog_images_in_octave[0].shape

    for attempt_index in range(num_attempts_until_convergence):
        # Convert from uint8 to float32 to compute derivatives and rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[
            image_index - 1 : image_index + 2
        ]
        pixel_cube = (
            np.stack(
                [
                    first_image[
                        row_index - 1 : row_index + 2, col_index - 1 : col_index + 2
                    ],
                    second_image[
                        row_index - 1 : row_index + 2, col_index - 1 : col_index + 2
                    ],
                    third_image[
                        row_index - 1 : row_index + 2, col_index - 1 : col_index + 2
                    ],
                ]
            ).astype("float32")
            / 255.0
        )

        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)

        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if all(abs(extremum_update) < 0.5):
            break

        col_index += int(round(extremum_update[0]))
        row_index += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))

        # Ensure the new pixel_cube lies entirely within the image
        if (
            row_index < image_border_width
            or row_index >= image_shape[0] - image_border_width
            or col_index < image_border_width
            or col_index >= image_shape[1] - image_border_width
            or image_index < 1
            or image_index > num_intervals
        ):
            is_extremum_outside_image = True
            break

    if is_extremum_outside_image or attempt_index >= num_attempts_until_convergence - 1:
        return None

    function_value_at_updated_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(
        gradient, extremum_update
    )

    if abs(function_value_at_updated_extremum) * num_intervals < contrast_threshold:
        return None

    xy_hessian = hessian[:2, :2]
    xy_hessian_trace = np.trace(xy_hessian)
    xy_hessian_det = np.linalg.det(xy_hessian)

    if (
        xy_hessian_det <= 0
        or eigenvalue_ratio * (xy_hessian_trace**2)
        >= ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det
    ):
        return None

    # Construct and return OpenCV KeyPoint object
    keypoint = cv2.KeyPoint()
    keypoint.pt = (
        (col_index + extremum_update[0]) * (2**octave_index),
        (row_index + extremum_update[1]) * (2**octave_index),
    )
    keypoint.octave = (
        octave_index
        + image_index * (2**8)
        + int(round((extremum_update[2] + 0.5) * 255)) * (2**16)
    )
    keypoint.size = (
        sigma
        * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals)))
        * (2 ** (octave_index + 1))
    )
    keypoint.response = abs(function_value_at_updated_extremum)
    return keypoint, image_index


#############################################################################################################
def computeGradientAtCenterPixel(array_of_pixels):
    """
        Approximate gradient at the center pixel [1, 1, 1] of a 3x3x3 array using central difference formula of order O(h^2),
        where h is the step size.

        Args:
        - array_of_pixels
    : A 3x3x3 NumPy array representing a neighborhood of pixels.

        Returns:
        - gradient: A 1D NumPy array representing the computed gradient [dx, dy, ds].
    """
    dx = 0.5 * (array_of_pixels[1, 1, 2] - array_of_pixels[1, 1, 0])
    dy = 0.5 * (array_of_pixels[1, 2, 1] - array_of_pixels[1, 0, 1])
    ds = 0.5 * (array_of_pixels[2, 1, 1] - array_of_pixels[0, 1, 1])
    return np.array(
        [dx, dy, ds]
    )  #############################################################################################################


def computeHessianAtCenterPixel(array_of_pixels):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size"""
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = array_of_pixels[1, 1, 1]
    dxx = array_of_pixels[1, 1, 2] - 2 * center_pixel_value + array_of_pixels[1, 1, 0]
    dyy = array_of_pixels[1, 2, 1] - 2 * center_pixel_value + array_of_pixels[1, 0, 1]
    dss = array_of_pixels[2, 1, 1] - 2 * center_pixel_value + array_of_pixels[0, 1, 1]
    dxy = 0.25 * (
        array_of_pixels[1, 2, 2]
        - array_of_pixels[1, 2, 0]
        - array_of_pixels[1, 0, 2]
        + array_of_pixels[1, 0, 0]
    )
    dxs = 0.25 * (
        array_of_pixels[2, 1, 2]
        - array_of_pixels[2, 1, 0]
        - array_of_pixels[0, 1, 2]
        + array_of_pixels[0, 1, 0]
    )
    dys = 0.25 * (
        array_of_pixels[2, 2, 1]
        - array_of_pixels[2, 0, 1]
        - array_of_pixels[0, 2, 1]
        + array_of_pixels[0, 0, 1]
    )
    return array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])


#############################################################################################################
def computeKeypointsWithOrientations(
    keypoint,
    octave_index,
    gaussian_image,
    radius_factor=3,
    num_bins=36,
    peak_ratio=0.8,
    scale_factor=1.5,
):
    """Compute orientations for each keypoint"""
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale_factor_multiplier = (
        scale_factor * keypoint.size / float32(2 ** (octave_index + 1))
    )
    radius_factor_multiplier = int(round(radius_factor * scale_factor_multiplier))
    weight_factor = -0.5 / (scale_factor_multiplier**2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius_factor_multiplier, radius_factor_multiplier + 1):
        region_y = int(round(keypoint.pt[1] / float32(2**octave_index))) + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius_factor_multiplier, radius_factor_multiplier + 1):
                region_x = int(round(keypoint.pt[0] / float32(2**octave_index))) + j
                if 0 < region_x < image_shape[1] - 1:
                    dx = (
                        gaussian_image[region_y, region_x + 1]
                        - gaussian_image[region_y, region_x - 1]
                    )
                    dy = (
                        gaussian_image[region_y - 1, region_x]
                        - gaussian_image[region_y + 1, region_x]
                    )
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i**2 + j**2))
                    histogram_index = int(
                        round(gradient_orientation * num_bins / 360.0)
                    )
                    raw_histogram[histogram_index % num_bins] += (
                        weight * gradient_magnitude
                    )

    for n in range(num_bins):
        smooth_histogram[n] = (
            6 * raw_histogram[n]
            + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins])
            + raw_histogram[n - 2]
            + raw_histogram[(n + 2) % num_bins]
        ) / 16.0

    max_orientation_value = max(smooth_histogram)
    orientation_peaks = where(
        logical_and(
            smooth_histogram > roll(smooth_histogram, 1),
            smooth_histogram > roll(smooth_histogram, -1),
        )
    )[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * max_orientation_value:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (
                peak_index
                + 0.5
                * (left_value - right_value)
                / (left_value - 2 * peak_value + right_value)
            ) % num_bins
            orientation = 360.0 - interpolated_peak_index * 360.0 / num_bins
            if abs(orientation - 360.0) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(
                *keypoint.pt,
                keypoint.size,
                orientation,
                keypoint.response,
                keypoint.octave
            )
            keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations


#############################################################################################################
def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2"""
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave

    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints"""
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if (
            last_unique_keypoint.pt[0] != next_keypoint.pt[0]
            or last_unique_keypoint.pt[1] != next_keypoint.pt[1]
            or last_unique_keypoint.size != next_keypoint.size
            or last_unique_keypoint.angle != next_keypoint.angle
        ):
            unique_keypoints.append(next_keypoint)
    return unique_keypoints


def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size"""
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale
#############################################################################################################
def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')



#############################################################################################################


image1 = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# keypoints1, descriptors1 = SIFT(image1)
# keypoints2, descriptors2 = SIFT(image2)

# print("keypoints done!")
# image1_with_keypoints = cv2.drawKeypoints(
#     image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# image2_with_keypoints = cv2.drawKeypoints(
#     image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )


#############################################################################################################
def matching(descriptor1 , descriptor2 , match_calculator):
    
    keypoints1 = descriptor1.shape[0]
    keypoints2 = descriptor2.shape[0]
    matches = []

    for kp1 in range(keypoints1):

        distance = -np.inf
        y_index = -1
        for kp2 in range(keypoints2):

         
            value = match_calculator(descriptor1[kp1], descriptor2[kp2])

            if value > distance:
              distance = value
              y_index = kp2
        
        match = cv2.DMatch()
        match.queryIdx = kp1
        match.trainIdx = y_index
        match.distance = distance
        matches.append(match)
    matches= sorted(matches, key=lambda x: x.distance, reverse=True)
    return matches
############################################################################################################# 

def calculate_ncc(descriptor1 , descriptor2):


    out1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1))
    out2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2))

    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    correlation = float(np.mean(correlation_vector))

    return correlation
#############################################################################################################

def calculate_ssd(descriptor1 , descriptor2):

    ssd = 0
    for m in range(len(descriptor1)):
        ssd += (descriptor1[m] - descriptor2[m]) ** 2

    ssd = - (np.sqrt(ssd))
    return ssd

#############################################################################################################

def get_matching(img1,img2,method):

    # Compute SIFT keypoints and descriptors
    start_time =time.time()
    keypoints_1, descriptor1 = SIFT(img1)

    keypoints_2, descriptor2 = SIFT(img2)

    end_time = time.time()
    Duration_sift = end_time - start_time

    if method  == 'ncc':
        start = time.time()
        matches_ncc = matching(descriptor1, descriptor2, calculate_ncc)
        matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                        matches_ncc[:30], img2, flags=2)
        end = time.time()
        match_time = end - start

    else:

        start = time.time()

        matches_ssd = matching(descriptor1, descriptor2, calculate_ssd)
        matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                        matches_ssd[:30], img2, flags=2)
        end = time.time()
        match_time = end - start
    
    return matched_image , match_time
# #############################################################################################################
matched_image, match_time = get_matching(image1, image2, "ssd")
print(match_time)
# Display the matched image
cv2.imshow("Matched Image", matched_image)
# cv2.imshow("Matched Image_kp1", image1_with_keypoints)
# cv2.imshow("Matched Image_kp2", image2_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()