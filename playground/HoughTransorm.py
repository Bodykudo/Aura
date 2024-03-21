import cv2
import numpy as np

class HoughTransform:
    @staticmethod
    def find_lines(edges, rho=1, theta=np.pi/180, threshold=100):
        height, width = edges.shape
        max_rho = int(np.sqrt(height ** 2 + width ** 2))
        rhos = np.arange(-max_rho, max_rho + 1, rho)
        thetas = np.deg2rad(np.arange(0, 180, theta))
        num_thetas = len(thetas)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        accumulator = np.zeros((2 * len(rhos), num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(edges)
        for i in range(num_thetas):
            rho_vals = x_idxs * cos_thetas[i] + y_idxs * sin_thetas[i]
            rho_vals += max_rho
            rho_vals = rho_vals.astype(np.int64)
            np.add.at(accumulator[:, i], rho_vals, 1)
        candidate_idxs = np.argwhere(accumulator >= threshold)
        candidate_values = accumulator[candidate_idxs[:, 0], candidate_idxs[:, 1]]
        sorted_idxs = np.argsort(candidate_values)[::-1][: len(candidate_values)]
        candidate_idxs = candidate_idxs[sorted_idxs]
        return candidate_idxs,rhos,thetas
    @staticmethod
    def draw_lines(candidate_idxs,rhos,thetas,image):
        for rho_idx, theta_idx in candidate_idxs:
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
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        return image
    @staticmethod
    def detect_lines(image, rho=1, theta=np.pi/180, threshold=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        candidate_idxs,rhos,thetas=HoughTransform.find_lines(edges,rho,theta,threshold)
        result=HoughTransform.draw_lines(candidate_idxs,rhos,thetas,image)
        return result


    # @staticmethod
    # def find_circles(edges,min_radius,max_radius,threshold):
    #     height, width = edges.shape
    #     accumulator = np.zeros((height, width, max_radius - min_radius + 1), dtype=np.uint64)
    #     for y in range(height):
    #         for x in range(width):
    #             if edges[y, x] != 0:
    #                 for r in range(min_radius, max_radius + 1):
    #                     for theta in range(0, 360):
    #                         a = x - int(r * np.cos(np.deg2rad(theta)))
    #                         b = y - int(r * np.sin(np.deg2rad(theta)))
    #                         if 0 <= a < width and 0 <= b < height:
    #                             accumulator[b, a, r - min_radius] += 1
    #     circles = np.argwhere(accumulator >= threshold)
    #     detected_circles = []
    #     for y, x, r_idx in circles:
    #         detected_circles.append((x, y, r_idx + min_radius))
    #     print(detected_circles.shape)
    #     return detected_circles
    # @staticmethod
    # def draw_circles(detected_circles):
    #     for x, y, r in detected_circles:
    #         cv2.circle(image, (x, y), r, (255, 255, 0), 2)
    #
    # @staticmethod
    # def detect_circles(image,min_radius=10,max_radius=100,threshold=30):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(gray, 50, 150)
    #     detected_circles=HoughTransform.find_circles(edges,min_radius,max_radius,threshold)
    #     # HoughTransform.draw_circles(detected_circles)


    @staticmethod
    def detect_ellipses():
        pass


# Testing
image = cv2.imread("lines.png")
lines_detected = HoughTransform.detect_lines(image)
cv2.imshow("Detected Lines", lines_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Testing Circle
# image = cv2.imread("circletest.jpg")
# lines_detected = HoughTransform.detect_circles(image)
# cv2.imshow("Detected Circle", lines_detected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
