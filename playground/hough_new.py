import cv2
import numpy as np


def detect_lines(image, rho_resolution, theta_resolution, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Compute the Hough Transform
    height, width = edges.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-max_rho, max_rho + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(0, 180, theta_resolution))
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

    # Find the peaks in the accumulator
    candidate_idxs = np.argwhere(accumulator >= threshold)
    candidate_values = accumulator[candidate_idxs[:, 0], candidate_idxs[:, 1]]
    sorted_idxs = np.argsort(candidate_values)[::-1][: len(candidate_values)]
    candidate_idxs = candidate_idxs[sorted_idxs]

    # Draw lines corresponding to the peaks
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


import cv2
import numpy as np


def preprocess_for_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges


def detect_circles(
    edges,
    dp=1,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100,
    circle_color=(255, 0, 255),
    circle_thickness=2,
):
    # Initialize accumulator for circle centers
    accumulator = np.zeros_like(edges)

    # Iterate over edge pixels
    edge_coords = np.argwhere(edges != 0)
    for y, x in edge_coords:
        # Vote for possible circle centers
        for r in range(minRadius, maxRadius + 1):
            a = b = 0
            # Vote for possible circle centers
            for theta in range(0, 360):
                a = int(x - r * np.cos(np.deg2rad(theta)))
                b = int(y - r * np.sin(np.deg2rad(theta)))
                if a >= 0 and a < edges.shape[1] and b >= 0 and b < edges.shape[0]:
                    accumulator[b, a] += 1  # Increment accumulator

    # Find potential circle centers
    centers = np.argwhere(accumulator >= param2)

    # Initialize output image
    output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Iterate through potential centers and draw circles
    for center_y, center_x in centers:
        cv2.circle(
            output_image, (center_x, center_y), 1, (0, 100, 100), 3
        )  # Draw center
        for r in range(minRadius, maxRadius + 1):
            # Generate circle coordinates
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = np.round(center_x + r * np.cos(theta)).astype(int)
            circle_y = np.round(center_y + r * np.sin(theta)).astype(int)

            # Check if all points in circle are within image bounds
            valid_indices = np.where(
                (circle_x >= 0)
                & (circle_x < edges.shape[1])
                & (circle_y >= 0)
                & (circle_y < edges.shape[0])
            )

            # Check if a circle at this center and radius is detected
            if (
                np.sum(edges[circle_y[valid_indices], circle_x[valid_indices]])
                >= param1
            ):
                cv2.circle(
                    output_image,
                    (center_x, center_y),
                    r,
                    circle_color,
                    circle_thickness,
                )  # Draw circle

    return output_image


def detect_ellipses(
    image,
    dp=1,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100,
    ellipse_color=(255, 0, 255),
    ellipse_thickness=2,
):
    edges = preprocess_for_hough(image)
    return detect_circles(
        edges,
        dp,
        minDist,
        param1,
        param2,
        minRadius,
        maxRadius,
        ellipse_color,
        ellipse_thickness,
    )


# Test the function
image = cv2.imread("image1.jpg")
ellipses_detected = detect_ellipses(
    image.copy(),
    dp=1,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100,
    ellipse_color=(255, 0, 255),
    ellipse_thickness=2,
)
cv2.imshow("Detected Ellipses", ellipses_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()
