import cv2
import numpy as np

from api.utils import read_image


class Hough:
    @staticmethod
    def calculate_accumulator(image, min_radius, max_radius):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        height, width = edges.shape
        accumulator = np.zeros(
            (height, width, max_radius - min_radius + 1), dtype=np.uint16
        )
        radius_range = np.arange(min_radius, max_radius + 1, 10)
        cos_values = np.cos(np.deg2rad(np.arange(0, 360, 10)))
        sin_values = np.sin(np.deg2rad(np.arange(0, 360, 10)))

        y, x = np.nonzero(edges)
        for r in radius_range:
            for theta in range(len(cos_values)):
                a = (x - r * cos_values[theta]).astype(int)
                b = (y - r * sin_values[theta]).astype(int)
                is_valid_mask = (a >= 0) & (a < width) & (b >= 0) & (b < height)
                accumulator[b[is_valid_mask], a[is_valid_mask], r - min_radius] += 1

        return accumulator

    @staticmethod
    def find_circles(accumulator, min_radius, max_radius, threshold_value=0.8):
        circles = []
        for r in range(max_radius - min_radius + 1):
            threshold = threshold_value * np.max(accumulator[:, :, r])
            y_index, x_index = np.nonzero(accumulator[:, :, r] > threshold)
            for x, y in zip(x_index, y_index):
                is_similar = False
                for circle in circles:
                    if (
                        np.linalg.norm(np.array(circle[:2]) - np.array([x, y]))
                        < circle[2] + r + min_radius
                    ):
                        is_similar = True
                        break
                if not is_similar:
                    circles.append((x, y, r + min_radius))
        return circles

    @staticmethod
    def draw_circles(image, circles, color):
        for circle in circles:
            cv2.circle(image, (circle[0], circle[1]), circle[2], color, 2)
        return image

    @staticmethod
    def detect_circles(
        image_path, min_radius, max_radius, threshold=80, color=(0, 0, 255)
    ):
        image = read_image(image_path)
        threshold_value = threshold / 100
        accumulator = Hough.calculate_accumulator(image, min_radius, max_radius)
        circles = Hough.find_circles(
            accumulator, min_radius, max_radius, threshold_value
        )
        output = Hough.draw_circles(image, circles, color)
        return output

    @staticmethod
    def find_lines(edges, rho=1, theta=np.pi / 180, threshold=100):
        height, width = edges.shape
        max_rho = int(np.sqrt(height**2 + width**2))
        rhos = np.arange(-max_rho, max_rho + 1, rho)
        thetas = np.deg2rad(np.arange(0, 180, theta))
        num_thetas = len(thetas)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        accumulator = np.zeros((2 * len(rhos), num_thetas), dtype=np.uint64)
        y_indices, x_indices = np.nonzero(edges)
        for i in range(num_thetas):
            rho_values = x_indices * cos_thetas[i] + y_indices * sin_thetas[i]
            rho_values += max_rho
            rho_values = rho_values.astype(np.int64)
            np.add.at(accumulator[:, i], rho_values, 1)
        cnadidates_indices = np.argwhere(accumulator >= threshold)
        candidate_values = accumulator[
            cnadidates_indices[:, 0], cnadidates_indices[:, 1]
        ]
        sorted_indices = np.argsort(candidate_values)[::-1][: len(candidate_values)]
        cnadidates_indices = cnadidates_indices[sorted_indices]
        return cnadidates_indices, rhos, thetas

    @staticmethod
    def draw_lines(image, cnadidates_indices, rhos, thetas, color):
        for rho_idx, theta_idx in cnadidates_indices:
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
        return image

    @staticmethod
    def detect_lines(image_path, rho=1, theta=180, threshold=100, color=(0, 0, 255)):
        image = read_image(image_path)
        grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscaled_image, 50, 150, apertureSize=3)
        candidates_indices, rhos, thetas = Hough.find_lines(
            edges, rho, np.pi / theta, threshold
        )
        result = Hough.draw_lines(image, candidates_indices, rhos, thetas, color)
        return result

    @staticmethod
    def detect_ellipses():
        pass