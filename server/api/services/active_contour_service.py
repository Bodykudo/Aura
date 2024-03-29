import numpy as np
import cv2

from api.utils import read_image


class ActiveContourService:
    @staticmethod
    def calculate_internal_energy(point, previous_point, next_point, alpha):
        dx1 = point[0] - previous_point[0]
        dy1 = point[1] - previous_point[1]
        dx2 = next_point[0] - point[0]
        dy2 = next_point[1] - point[1]
        denominator = pow(dx1 * dx1 + dy1 * dy1, 1.5)
        curvature = 0 if denominator == 0 else (dx1 * dy2 - dx2 * dy1) / denominator
        return alpha * curvature

    @staticmethod
    def calculate_external_energy(image, point, beta):
        x = max(0, min(point[0], image.shape[1] - 1))
        y = max(0, min(point[1], image.shape[0] - 1))
        return -beta * image[y, x]

    @staticmethod
    def calculate_gradients(point, prev_point, gamma):
        dx = point[0] - prev_point[0]
        dy = point[1] - prev_point[1]
        return gamma * (dx * dx + dy * dy)

    @staticmethod
    def calculate_point_energy(
        image, point, prev_point, next_point, alpha, beta, gamma
    ):
        internal_energy = ActiveContourService.calculate_internal_energy(
            point, prev_point, next_point, alpha
        )
        external_energy = ActiveContourService.calculate_external_energy(
            image, point, beta
        )
        gradients = ActiveContourService.calculate_gradients(point, prev_point, gamma)
        return internal_energy + external_energy + gradients

    @staticmethod
    def snake_operation(image, curve, window_size, alpha, beta, gamma):
        new_curve = []
        window_index = (window_size - 1) // 2
        num_points = len(curve)

        for i in range(num_points):
            pt = curve[i]
            prev_pt = curve[(i - 1 + num_points) % num_points]
            next_pt = curve[(i + 1) % num_points]
            min_energy = float("inf")
            new_pt = pt

            for dx in range(-window_index, window_index + 1):
                for dy in range(-window_index, window_index + 1):
                    move_pt = (pt[0] + dx, pt[1] + dy)
                    energy = ActiveContourService.calculate_point_energy(
                        image, move_pt, prev_pt, next_pt, alpha, beta, gamma
                    )
                    if energy < min_energy:
                        min_energy = energy
                        new_pt = move_pt
            new_curve.append(new_pt)

        return new_curve

    @staticmethod
    def initialize_contours(center, radius, number_of_points):
        curve = []
        current_angle = 0
        resolution = 360 / number_of_points

        for _ in range(number_of_points):
            x_p = int(center[0] + radius * np.cos(np.radians(current_angle)))
            y_p = int(center[1] + radius * np.sin(np.radians(current_angle)))
            current_angle += resolution
            curve.append((x_p, y_p))

        return curve

    @staticmethod
    def draw_contours(image, snake_points):
        output_image = image.copy()
        for i in range(len(snake_points)):
            cv2.circle(output_image, snake_points[i], 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(
                    output_image, snake_points[i - 1], snake_points[i], (255, 0, 0), 2
                )
        cv2.line(output_image, snake_points[0], snake_points[-1], (255, 0, 0), 2)
        return output_image

    @staticmethod
    def calculate_distance(point1, point2):
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    @staticmethod
    def calculate_contour_perimeter(curve):
        perimeter = 0
        num_points = len(curve)
        for i in range(num_points):
            point1 = curve[i]
            point2 = curve[(i + 1) % num_points]
            perimeter += ActiveContourService.calculate_distance(point1, point2)
        return perimeter

    @staticmethod
    def active_contour(
        image_path,
        center,
        radius,
        num_iterations,
        num_points,
        window_size,
        alpha,
        beta,
        gamma,
    ):
        image = read_image(image_path)
        test = image.copy()
        cv2.circle(test, center, radius, (0, 255, 0), 2)
        cv2.imshow("Original Image with Circle", test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        curve = ActiveContourService.initialize_contours(center, radius, num_points)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for _ in range(num_iterations):
            curve = ActiveContourService.snake_operation(
                gray_image, curve, window_size, alpha, beta, gamma
            )

        output = ActiveContourService.draw_contours(image, curve)
        perimeter = ActiveContourService.calculate_contour_perimeter(curve)

        return output, perimeter


# input_image = cv2.imread("tofaha.jpeg")
# output_image = input_image.copy()

# # Draw circle on the original image
# circle_center = (124, 101)  # Example center
# circle_radius = 120  # Example radius
# cv2.circle(output_image, circle_center, circle_radius, (0, 255, 0), 2)

# cv2.imshow("Original Image with Circle", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# num_iterations = 300
# num_points = 70
# window_size = 11
# alpha = 10
# beta = 3
# gamma = 1

# output_image, perimter = ActiveContourService.active_contour(
#     input_image,
#     circle_center,
#     circle_radius,
#     num_iterations,
#     num_points,
#     window_size,
#     alpha,
#     beta,
#     gamma,
# )


# cv2.imshow("Output Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(f"Perimeter: {perimter}")
