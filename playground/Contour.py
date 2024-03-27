import cv2
import numpy as np


class Contour:
    @staticmethod
    def find_contours(image):
        """
        This function implements a simplified contour finding algorithm.

        Args:
            image: A grayscale image represented as a NumPy array.

        Returns:
            A list of contours, where each contour is a list of pixels.
        """

        # Threshold the image to isolate potential object regions
        thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

        # Find connected components (potential objects)
        # We'll use iterative approach to explore connected pixels
        contours = []
        visited = np.zeros(thresh.shape, dtype=bool)  # Keep track of visited pixels

        for row in range(thresh.shape[0]):
            for col in range(thresh.shape[1]):
                if thresh[row, col] == 255 and not visited[row, col]:
                    # Found a starting point of a potential contour, explore neighbors
                    contour = Contour.explore_neighbors(thresh, visited, row, col)
                    contours.append(contour)

        return contours

    @staticmethod
    def explore_neighbors(image, visited, row, col):
        """
        This function explores connected white pixels (objects) recursively.

        Args:
            image: Binary image.
            visited: Visited pixels map.
            row: Current pixel row.
            col: Current pixel col.

        Returns:
            A list of pixels representing the contour.
        """
        visited[row, col] = True  # Mark current pixel as visited
        contour = [(row, col)]  # Add current pixel to contour

        # Explore neighbors recursively in 4 directions (up, down, left, right)
        if row > 0 and image[row - 1, col] == 255 and not visited[row - 1, col]:
            contour.extend(Contour.explore_neighbors(image, visited, row - 1, col))
        if (
            row < image.shape[0] - 1
            and image[row + 1, col] == 255
            and not visited[row + 1, col]
        ):
            contour.extend(Contour.explore_neighbors(image, visited, row + 1, col))
        if col > 0 and image[row, col - 1] == 255 and not visited[row, col - 1]:
            contour.extend(Contour.explore_neighbors(image, visited, row, col - 1))
        if (
            col < image.shape[1] - 1
            and image[row, col + 1] == 255
            and not visited[row, col + 1]
        ):
            contour.extend(Contour.explore_neighbors(image, visited, row, col + 1))

        return contour

    @staticmethod
    def draw_contours(image, contours, color=(0, 255, 0), thickness=2):
        """
        This simplified function draws contours on an image.

        Args:
            image: A NumPy array representing the image.
            contours: A list of contours, where each contour is a list of pixels.
            color: The color of the contours (BGR format). Defaults to green.
            thickness: The thickness of the contour lines. Defaults to 2.

        Returns:
            The image with contours drawn on it.
        """
        output_image = (
            image.copy()
        )  # Create a copy to avoid modifying the original image

        for cnt in contours:
            # Convert contour to NumPy array for compatibility with OpenCV functions
            cnt_array = np.array(cnt, dtype=np.int32)

            # Draw the contour on the image using cv2.polylines
            cv2.polylines(output_image, [cnt_array], True, color, thickness)

        return output_image

    def apply_contour(image_path: str):
        image = cv2.imread(image_path)
        # Convert the image to grayscale
        grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(grayscaled_image, (5, 5), 0)

        # Use Canny edge detection to find edges in the image
        edges = cv2.Canny(blurred_image, 30, 150)

        # Use your custom find_contours function
        contours = Contour.find_contours(edges)

        transformed_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        transformed_image = cv2.flip(transformed_image, 0)
        output = Contour.draw_contours(transformed_image, contours)

        output = cv2.flip(output, 0)
        output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        return output


output = Contour.apply_contour("image1.jpg")
cv2.imshow("Contours", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
