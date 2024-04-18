import cv2
import numpy as np

from api.utils import read_image


class CornersDetector:
    @staticmethod
    def harris_corner_detection(
        image_path: str,
        block_size: int = 2,
        kernel_size: int = 3,
        k: float = 0.04,
        threshold: float = 0.01,
    ):
        image = read_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        dx_squared = dx**2
        dy_squared = dy**2
        dxy = dx * dy

        kernel = np.ones((block_size, block_size), dtype=np.float64) / (block_size**2)

        sum_dx_squared = cv2.filter2D(dx_squared, -1, kernel)
        sum_dy_squared = cv2.filter2D(dy_squared, -1, kernel)
        sum_dxy = cv2.filter2D(dxy, -1, kernel)

        det_M = (sum_dx_squared * sum_dy_squared) - (sum_dxy**2)
        trace_M = sum_dx_squared + sum_dy_squared

        harris_response = det_M - k * (trace_M**2)

        corners = np.argwhere(harris_response > threshold * harris_response.max())
        for corner in corners:
            y, x = corner
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        return image

    @staticmethod
    def lambda_corner_detection(
        image_path: str,
        kernel_size: int = 3,
        max_corners: int = 25,
        quality_level: float = 0.01,
        min_distance: int = 10,
    ):
        image = read_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        corner_mask = np.zeros_like(gradient_magnitude, dtype=np.uint8)
        local_maxima = (
            cv2.dilate(gradient_magnitude, np.ones((3, 3))) == gradient_magnitude
        )
        corner_mask[local_maxima] = 1

        corner_scores = corner_mask * gradient_magnitude
        corner_indices = np.unravel_index(
            np.argsort(corner_scores.ravel())[::-1][:max_corners], corner_scores.shape
        )
        corners = np.column_stack(corner_indices)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)
        return image

    @staticmethod
    def detect_corners(image_path, harris_params={}, lambda_params={}):
        if not lambda_params:
            return CornersDetector.harris_corner_detection(image_path, **harris_params)

        if not harris_params:
            return CornersDetector.lambda_corner_detection(image_path, **lambda_params)

        harris_image = CornersDetector.harris_corner_detection(
            image_path, **harris_params
        )
        lambda_image = CornersDetector.lambda_corner_detection(
            image_path, **lambda_params
        )
        return cv2.addWeighted(harris_image, 0.5, lambda_image, 0.5, 0)
