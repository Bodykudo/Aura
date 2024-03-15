import cv2
import numpy as np


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(
        edges, 1, np.pi / 180, 100
    )  # Decrease threshold for better line detection

    if lines is not None:
        for rho, theta in lines[:, 0]:
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


def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100,
    )  # Adjust parameters for better circle detection

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    return image


def detect_ellipses(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    ellipses = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100,
    )  # Adjust parameters for better ellipse detection

    if ellipses is not None:
        ellipses = np.uint16(np.around(ellipses))
        for i in ellipses[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    return image


# Test the functions
image = cv2.imread("image1.jpg")

# Detect lines
# lines_detected = detect_lines(image.copy())
# cv2.imshow("Detected Lines", lines_detected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Detect circles
circles_detected = detect_circles(image.copy())
cv2.imshow("Detected Circles", circles_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect ellipses
# ellipses_detected = detect_ellipses(image.copy())
# cv2.imshow("Detected Ellipses", ellipses_detected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
