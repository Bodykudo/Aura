import cv2
import numpy as np
from functools import cmp_to_key

from api.utils import read_image

FLOAT_TOLERANCE = 1e-7

class SIFT:
    def __init__(
        self,
        image_path,
        sigma=1.6,
        num_intervals=3,
        assumed_blur=0.5,
    ):
        self.image = read_image(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = self.gray.astype("float32")
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.assumed_blur = assumed_blur
        self.image_border_width = 5
        self.gaussian_images = []
        self.keypoints = []
        self.descriptors = []

    def apply(self):
        self._detect_keypoints()
        self._detect_descriptors()

    def get_keypoints(self):
        return self.keypoints

    def get_descriptors(self):
        return self.descriptors

    def draw_keypoints(self):
        return cv2.drawKeypoints(
            self.image,
            self.keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    def _detect_keypoints(self):
        base_image = self._create_modified_image()
        octaves_count = self._calculate_octaves_count(base_image.shape)
        gaussian_kernels = self._get_gaussian_filters_sizes()
        self.gaussian_images = self._get_pyramids(
            base_image, octaves_count, gaussian_kernels
        )
        dog_images = self._get_dog(self.gaussian_images)
        keypoints = self._detect_scale_space_extrema(
            self.gaussian_images,
            dog_images,
            self.num_intervals,
            self.sigma,
            self.image_border_width,
        )
        keypoints = self._remove_duplicate_keypoints(keypoints)
        self.keypoints = self._convert_keypoints_to_input_image_size(keypoints)

    def _detect_descriptors(self):
        self.descriptors = self._generate_descriptors(
            self.keypoints, self.gaussian_images
        )

    def _create_modified_image(self):
        modified_image = cv2.resize(
            self.gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR
        )
        adjusted_sigma = np.sqrt(
            max((self.sigma**2) - ((2 * self.assumed_blur) ** 2), 0.01)
        )
        return cv2.GaussianBlur(
            modified_image, (0, 0), sigmaX=adjusted_sigma, sigmaY=adjusted_sigma
        )

    def _calculate_octaves_count(self, image_dimensions):
        return int(np.round(np.log(min(image_dimensions)) / np.log(2) - 1))

    def _get_gaussian_filters_sizes(self):
        images_per_octave = self.num_intervals + 3
        k_value = 2 ** (1.0 / self.num_intervals)
        gaussian_filter_sizes = np.zeros(images_per_octave)
        gaussian_filter_sizes[0] = self.sigma

        for index in range(1, images_per_octave):
            previous_sigma = (k_value ** (index - 1)) * self.sigma
            total_sigma = k_value * previous_sigma
            gaussian_filter_sizes[index] = np.sqrt(total_sigma**2 - previous_sigma**2)
        return gaussian_filter_sizes

    def _get_pyramids(self, image_data, octave_count, gaussian_filters):
        gaussian_pyramid = []
        for octave_index in range(octave_count):
            octave_images = []
            octave_images.append(image_data)
            for filter_size in gaussian_filters[1:]:
                image_data = cv2.GaussianBlur(
                    image_data, (0, 0), sigmaX=filter_size, sigmaY=filter_size
                )
                octave_images.append(image_data)
            gaussian_pyramid.append(octave_images)
            base_image = octave_images[-3]
            image_data = cv2.resize(
                base_image,
                (int(base_image.shape[1] / 2), int(base_image.shape[0] / 2)),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.array(gaussian_pyramid, dtype=object)

    def _get_dog(self, pyramids):
        dog_pyramid = []

        for octave_images in pyramids:
            dog_images_in_octave = []
            for first_img, second_img in zip(octave_images, octave_images[1:]):
                dog_images_in_octave.append(cv2.subtract(second_img, first_img))
            dog_pyramid.append(dog_images_in_octave)
        return np.array(dog_pyramid, dtype=object)

    def _detect_scale_space_extrema(
        self,
        gaussian_pyramid,
        dog_pyramid,
        interval_count,
        sigma_value,
        border_width,
        contrast_threshold=0.04,
    ):
        threshold_value = np.floor(0.5 * contrast_threshold / interval_count * 255)
        found_keypoints = []

        for octave_index, dog_octave in enumerate(dog_pyramid):
            for image_index, (first_img, second_img, third_img) in enumerate(
                zip(dog_octave, dog_octave[1:], dog_octave[2:])
            ):
                for x in range(border_width, first_img.shape[0] - border_width):
                    for y in range(border_width, first_img.shape[1] - border_width):
                        if self._check_pixel_extremum(
                            first_img[x - 1 : x + 2, y - 1 : y + 2],
                            second_img[x - 1 : x + 2, y - 1 : y + 2],
                            third_img[x - 1 : x + 2, y - 1 : y + 2],
                            threshold_value,
                        ):
                            extremum_result = self._localize_extremum_via_quadratic_fit(
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
                                    self._compute_keypoints_with_orientations(
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

    def _check_pixel_extremum(
        self, first_subimage, second_subimage, third_subimage, threshold
    ):
        center_pixel_value = second_subimage[1, 1]

        if abs(center_pixel_value) <= threshold:
            return False

        if center_pixel_value > 0:
            return (
                np.all(center_pixel_value >= first_subimage)
                and np.all(center_pixel_value >= third_subimage)
                and np.all(center_pixel_value >= second_subimage[0, :])
                and np.all(center_pixel_value >= second_subimage[2, :])
                and center_pixel_value >= second_subimage[1, 0]
                and center_pixel_value >= second_subimage[1, 2]
            )
        elif center_pixel_value < 0:
            return (
                np.all(center_pixel_value <= first_subimage)
                and np.all(center_pixel_value <= third_subimage)
                and np.all(center_pixel_value <= second_subimage[0, :])
                and np.all(center_pixel_value <= second_subimage[2, :])
                and center_pixel_value <= second_subimage[1, 0]
                and center_pixel_value <= second_subimage[1, 2]
            )

    def _localize_extremum_via_quadratic_fit(
        self,
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
        is_extremum_outside_image = False
        image_shape = dog_images_in_octave[0].shape

        for attempt_index in range(num_attempts_until_convergence):
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

            gradient = self._compute_gradient_at_center_pixel(pixel_cube)
            hessian = self._compute_hessian_at_center_pixel(pixel_cube)

            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            if np.all(abs(extremum_update) < 0.5):
                break

            col_index += int(np.round(extremum_update[0]))
            row_index += int(np.round(extremum_update[1]))
            image_index += int(np.round(extremum_update[2]))

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

        if (
            is_extremum_outside_image
            or attempt_index >= num_attempts_until_convergence - 1
        ):
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

        keypoint = cv2.KeyPoint()
        keypoint.pt = (
            (col_index + extremum_update[0]) * (2**octave_index),
            (row_index + extremum_update[1]) * (2**octave_index),
        )
        keypoint.octave = (
            octave_index
            + image_index * (2**8)
            + int(np.round((extremum_update[2] + 0.5) * 255)) * (2**16)
        )
        keypoint.size = (
            sigma
            * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals)))
            * (2 ** (octave_index + 1))
        )
        keypoint.response = abs(function_value_at_updated_extremum)
        return keypoint, image_index

    def _compute_gradient_at_center_pixel(self, array_of_pixels):
        dx = 0.5 * (array_of_pixels[1, 1, 2] - array_of_pixels[1, 1, 0])
        dy = 0.5 * (array_of_pixels[1, 2, 1] - array_of_pixels[1, 0, 1])
        ds = 0.5 * (array_of_pixels[2, 1, 1] - array_of_pixels[0, 1, 1])
        return np.array([dx, dy, ds])

    def _compute_hessian_at_center_pixel(self, array_of_pixels):
        center_pixel_value = array_of_pixels[1, 1, 1]
        dxx = (
            array_of_pixels[1, 1, 2] - 2 * center_pixel_value + array_of_pixels[1, 1, 0]
        )
        dyy = (
            array_of_pixels[1, 2, 1] - 2 * center_pixel_value + array_of_pixels[1, 0, 1]
        )
        dss = (
            array_of_pixels[2, 1, 1] - 2 * center_pixel_value + array_of_pixels[0, 1, 1]
        )
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
        return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

    def _compute_keypoints_with_orientations(
        self,
        keypoint,
        octave_index,
        gaussian_image,
        radius_factor=3,
        num_bins=36,
        peak_ratio=0.8,
        scale_factor=1.5,
    ):
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale_factor_multiplier = (
            scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        )
        radius_factor_multiplier = int(
            np.round(radius_factor * scale_factor_multiplier)
        )
        weight_factor = -0.5 / (scale_factor_multiplier**2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius_factor_multiplier, radius_factor_multiplier + 1):
            region_y = int(np.round(keypoint.pt[1] / np.float32(2**octave_index))) + i
            if 0 < region_y < image_shape[0] - 1:
                for j in range(-radius_factor_multiplier, radius_factor_multiplier + 1):
                    region_x = (
                        int(np.round(keypoint.pt[0] / np.float32(2**octave_index))) + j
                    )
                    if 0 < region_x < image_shape[1] - 1:
                        dx = (
                            gaussian_image[region_y, region_x + 1]
                            - gaussian_image[region_y, region_x - 1]
                        )
                        dy = (
                            gaussian_image[region_y - 1, region_x]
                            - gaussian_image[region_y + 1, region_x]
                        )
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i**2 + j**2))
                        histogram_index = int(
                            np.round(gradient_orientation * num_bins / 360.0)
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
        orientation_peaks = np.where(
            np.logical_and(
                smooth_histogram > np.roll(smooth_histogram, 1),
                smooth_histogram > np.roll(smooth_histogram, -1),
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
                if abs(orientation - 360.0) < FLOAT_TOLERANCE:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(
                    *keypoint.pt,
                    keypoint.size,
                    orientation,
                    keypoint.response,
                    keypoint.octave,
                )
                keypoints_with_orientations.append(new_keypoint)

        return keypoints_with_orientations

    def _compare_keypoints(self, keypoint1, keypoint2):
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

    def _remove_duplicate_keypoints(self, keypoints):
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self._compare_keypoints))
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

    def _convert_keypoints_to_input_image_size(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def _unpack_octave(self, keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale

    def _generate_descriptors(
        self,
        keypoints,
        gaussian_images,
        window_width=4,
        num_bins=8,
        scale_multiplier=3,
        descriptor_max_value=0.2,
    ):
        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self._unpack_octave(keypoint)
            gaussian_image = gaussian_images[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype("int")
            bins_per_degree = num_bins / 360.0
            angle = 360.0 - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(
                np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5)
            )
            half_width = int(min(half_width, np.sqrt(num_rows**2 + num_cols**2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if (
                        row_bin > -1
                        and row_bin < window_width
                        and col_bin > -1
                        and col_bin < window_width
                    ):
                        window_row = int(np.round(point[1] + row))
                        window_col = int(np.round(point[0] + col))
                        if (
                            window_row > 0
                            and window_row < num_rows - 1
                            and window_col > 0
                            and window_col < num_cols - 1
                        ):
                            dx = (
                                gaussian_image[window_row, window_col + 1]
                                - gaussian_image[window_row, window_col - 1]
                            )
                            dy = (
                                gaussian_image[window_row - 1, window_col]
                                - gaussian_image[window_row + 1, window_col]
                            )
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(
                                weight_multiplier
                                * (
                                    (row_rot / hist_width) ** 2
                                    + (col_rot / hist_width) ** 2
                                )
                            )
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append(
                                (gradient_orientation - angle) * bins_per_degree
                            )

            for row_bin, col_bin, magnitude, orientation_bin in zip(
                row_bin_list, col_bin_list, magnitude_list, orientation_bin_list
            ):
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor(
                    [row_bin, col_bin, orientation_bin]
                ).astype(int)
                row_fraction, col_fraction, orientation_fraction = (
                    row_bin - row_bin_floor,
                    col_bin - col_bin_floor,
                    orientation_bin - orientation_bin_floor,
                )
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

                histogram_tensor[
                    row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor
                ] += c000
                histogram_tensor[
                    row_bin_floor + 1,
                    col_bin_floor + 1,
                    (orientation_bin_floor + 1) % num_bins,
                ] += c001
                histogram_tensor[
                    row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor
                ] += c010
                histogram_tensor[
                    row_bin_floor + 1,
                    col_bin_floor + 2,
                    (orientation_bin_floor + 1) % num_bins,
                ] += c011
                histogram_tensor[
                    row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor
                ] += c100
                histogram_tensor[
                    row_bin_floor + 2,
                    col_bin_floor + 1,
                    (orientation_bin_floor + 1) % num_bins,
                ] += c101
                histogram_tensor[
                    row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor
                ] += c110
                histogram_tensor[
                    row_bin_floor + 2,
                    col_bin_floor + 2,
                    (orientation_bin_floor + 1) % num_bins,
                ] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), FLOAT_TOLERANCE)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype="float32")

    @staticmethod
    def match_features(
        original_image_descriptors, template_image_descriptors, matching_method
    ):
        original_image_keypoints = original_image_descriptors.shape[0]
        template_image_keypoints = template_image_descriptors.shape[0]

        matchings = []
        for teamplate_keypoint in range(template_image_keypoints):
            distance = -np.inf
            y_index = -1
            for original_keypoint in range(original_image_keypoints):
                value = matching_method(
                    template_image_descriptors[teamplate_keypoint],
                    original_image_descriptors[original_keypoint],
                )
                if value > distance:
                    distance = value
                    y_index = original_keypoint
            matching = cv2.DMatch()
            matching.queryIdx = teamplate_keypoint
            matching.trainIdx = y_index
            matching.distance = distance
            matchings.append(matching)
        matchings = sorted(matchings, key=lambda x: x.distance, reverse=True)
        return matchings

    @staticmethod
    def ncc_matching(descriptor_1, descriptor_2):
        out1_normalized = (descriptor_1 - np.mean(descriptor_1)) / (
            np.std(descriptor_1)
        )
        out2_normalized = (descriptor_2 - np.mean(descriptor_2)) / (
            np.std(descriptor_2)
        )
        correlation_vector = np.multiply(out1_normalized, out2_normalized)
        correlation = float(np.mean(correlation_vector))
        return correlation

    @staticmethod
    def ssd_matching(descriptor_1, descriptor_2):
        ssd = 0
        for m in range(len(descriptor_1)):
            ssd += (descriptor_1[m] - descriptor_2[m]) ** 2
        ssd = -(np.sqrt(ssd))
        return ssd

    @staticmethod
    def match_images(
        origina_image_path,
        original_image_keypoints,
        original_image_descriptors,
        template_image_path,
        template_image_keypoints,
        template_image_descriptors,
        method,
        num_matches=30,
    ):
        if method not in ["ncc", "ssd"]:
            raise ValueError("Invalid method")

        original_image = read_image(origina_image_path)
        template_image = read_image(template_image_path)

        if method == "ncc":
            matches_ncc = SIFT.match_features(
                original_image_descriptors,
                template_image_descriptors,
                SIFT.ncc_matching,
            )
            matched_image = cv2.drawMatches(
                template_image,
                template_image_keypoints,
                original_image,
                original_image_keypoints,
                matches_ncc[
                    : num_matches > len(matches_ncc) and len(matches_ncc) or num_matches
                ],
                original_image,
                flags=2,
            )
        else:
            matches_ssd = SIFT.match_features(
                original_image_descriptors,
                template_image_descriptors,
                SIFT.ssd_matching,
            )
            matched_image = cv2.drawMatches(
                template_image,
                template_image_keypoints,
                original_image,
                original_image_keypoints,
                matches_ssd[
                    : num_matches > len(matches_ssd) and len(matches_ssd) or num_matches
                ],
                original_image,
                flags=2,
            )
        return matched_image
