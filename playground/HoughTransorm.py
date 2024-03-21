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
    def draw_lines(candidate_idxs,rhos,thetas):
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
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image
    @staticmethod
    def detect_lines(image, rho=1, theta=np.pi/180, threshold=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        candidate_idxs,rhos,thetas=HoughTransform.find_lines(edges,rho,theta,threshold)
        result=HoughTransform.draw_lines(candidate_idxs,rhos,thetas)
        return result


    @staticmethod
    def detect_circles():
        pass
    @staticmethod
    def detect_ellipses():
        pass


# Testing
image = cv2.imread("test_lines.png")
lines_detected = HoughTransform.detect_lines(image)
cv2.imshow("Detected Lines", lines_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()