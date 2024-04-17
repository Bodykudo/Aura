import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_scale_space_pyramid(
    image, num_octaves=3, num_scales=5, sigma=1.6, downsampling_factor=2
):
    """
    Constructs a scale-space pyramid representation of a grayscale image.

    A scale-space pyramid is a multi-scale image representation that facilitates
    tasks like feature detection and object recognition. It comprises multiple
    octaves, each containing several scales. Images within a scale are progressively
    blurred versions of the original, capturing features at varying levels of detail.
    Images across octaves are downsampled versions of each other, enabling the
    detection of larger structures.

    Parameters:
        image (numpy.ndarray): The input grayscale image as a NumPy array.
        num_octaves (int, optional): The number of octaves in the pyramid.
            Each octave represents a doubling of the image size. Defaults to 3.
        num_scales (int, optional): The number of scales per octave. Each scale
            represents a level of Gaussian blurring applied to the image. Defaults to 5.
        sigma (float, optional): The initial standard deviation for the
            Gaussian blur kernel, controlling the level of smoothing. Defaults to 1.6.
        downsampling_factor (int, optional): The downsampling factor between
            octaves. Higher values lead to coarser scales but reduce computational
            cost. Defaults to 2 (downsample by half in each octave).

    Returns:
        list: A list of lists representing the scale-space pyramid. The outer list
            contains one sub-list for each octave, and each sub-list contains the
            images for each scale within that octave.
    """

    pyramid = []
    current_image = image.copy()  # Create a copy to avoid modifying the original

    for octave_level in range(num_octaves):
        print(f"Processing octave {octave_level + 1}...")
        octave = []

        for scale_level in range(num_scales):
            print(f"  Building scale {scale_level + 1}...")

            # Apply Gaussian blur for scale invariance
            blurred_image = cv2.GaussianBlur(current_image, (0, 0), sigma)
            octave.append(blurred_image)

            # Increase sigma for the next scale in the octave (effectively scaling blur)
            sigma *= 2.0  # Double sigma for the next scale level

        # Downsample the image for the next octave (reduce resolution)
        new_width = int(current_image.shape[1] / downsampling_factor)
        new_height = int(current_image.shape[0] / downsampling_factor)
        print(f"  Downscaling image to {new_width}x{new_height} for next octave")
        current_image = cv2.resize(
            current_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

        # Reset sigma for the next octave
        sigma = 1.6

        pyramid.append(octave)

    return pyramid


def generate_DoG_pyramid(gaussian_pyramid):
    """
    Generates the Difference-of-Gaussian (DoG) pyramid from the Gaussian pyramid.

    The DoG pyramid highlights image features that are robust across scales, making
    it valuable for tasks like keypoint detection.

    Parameters:
        gaussian_pyramid (list): The Gaussian scale-space pyramid.

    Returns:
        list: A list of lists representing the Difference-of-Gaussian pyramid.
    """

    DoG_pyramid = []
    for octave in gaussian_pyramid:
        DoG_octave = []
        for i in range(len(octave) - 1):
            # Calculate DoG by subtracting consecutive scales within an octave
            DoG_image = octave[i + 1] - octave[i]
            DoG_octave.append(DoG_image)
        DoG_pyramid.append(DoG_octave)

    return DoG_pyramid


# Testing
image_path = 'cat.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

gaussian_pyramid = build_scale_space_pyramid(image)

# Generate the Difference-of-Gaussian (DoG) pyramid
DoG_pyramid = generate_DoG_pyramid(gaussian_pyramid)

# # Display all levels of the pyramid (optional)
# for octave_level, octave in enumerate(gaussian_pyramid):
#     for scale_level, image_level in enumerate(octave):
#         print(f"Octave {octave_level + 1}, Scale level {scale_level + 1}:")
#         print(f"  Resolution: {image_level.shape[1]}x{image_level.shape[0]}")
#         print(f"  Scale: {1.6 * (2 ** scale_level)}\n")
#         cv2.imshow(f"Octave {octave_level + 1}, Scale {scale_level + 1}", image_level)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# # Display all levels of the DoG pyramid (optional)
# for octave_level, octave in enumerate(DoG_pyramid):
#     for scale_level, image_level in enumerate(octave):
#         print(f"Octave {octave_level + 1}, Scale level {scale_level + 1}:")
#         print(f"  Resolution: {image_level.shape[1]}x{image_level.shape[0]}")
#         cv2.imshow(
#             f"DoG Octave {octave_level + 1}, Scale {scale_level + 1}", image_level
#         )
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# print("Scale-space pyramid and DoG pyramid generation complete!")


def detect_keypoints(DoG_pyramid, threshold, edge_threshold=0.03):
    keypoints = []
    for octave, octave_images in enumerate(DoG_pyramid):
        for scale, img in enumerate(octave_images[1:-1]):  
            prev_img = octave_images[scale]
            next_img = octave_images[scale + 2]
            keypoints.extend(detect_keypoints_in_image(img, prev_img, next_img, threshold, edge_threshold,0))
    return keypoints

import numpy as np

def is_local_extremum(center, same_picture_neighbors, adjacent_picture_neighbors):
    center_val = center[1, 1]
    
    # Combine all neighboring pixels
    all_neighbors = np.concatenate((same_picture_neighbors, adjacent_picture_neighbors))
    
    # Check if the center pixel is a local extremum
    if (center_val > np.max(all_neighbors)) or (center_val < np.min(all_neighbors)):
        return True
    else:
        return False



def is_edge_point(img, i, j, edge_threshold):
    dx = img[i+1, j] - img[i-1, j]
    dy = img[i, j+1] - img[i, j-1]
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    if gradient_magnitude > edge_threshold:
        return True
    return False

def detect_keypoints_in_image(img, prev_img, next_img, threshold, edge_threshold, min_distance):
    keypoints = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if is_local_extremum(img[i-1:i+2, j-1:j+2], prev_img[i-1:i+2, j-1:j+2], next_img[i-1:i+2, j-1:j+2]):
                center_val = img[i, j]
                if abs(center_val) < threshold:
                    continue
                if is_edge_point(img, i, j, edge_threshold):
                    continue
                # Refinement step
                if refine_keypoint(img, i, j, threshold, edge_threshold):
                    keypoints.append((i, j))
    
    # Non-maximum suppression
    keypoints = sorted(keypoints, key=lambda x: img[x[0], x[1]], reverse=True)  # Sort by intensity
    filtered_keypoints = []
    for keypoint in keypoints:
        if not any(np.linalg.norm(np.array(keypoint) - np.array(existing)) < min_distance for existing in filtered_keypoints):
            filtered_keypoints.append(keypoint)
    
    return filtered_keypoints



def refine_keypoint(img, i, j, threshold, edge_threshold):
    # Compute Hessian matrix components
    Dxx = img[i, j + 1] + img[i, j - 1] - 2 * img[i, j]
    Dyy = img[i + 1, j] + img[i - 1, j] - 2 * img[i, j]
    Dxy = (img[i + 1, j + 1] - img[i + 1, j - 1] - img[i - 1, j + 1] + img[i - 1, j - 1]) / 4
    
    # Compute trace and determinant of Hessian matrix
    trace_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy ** 2
    
    # Calculate the ratio of the eigenvalues
    r = trace_H ** 2 / det_H
    
    # Check if the keypoint meets the criteria for refinement
    if det_H > 0.03 and (r+2) < threshold:
        return True
    else:
        return False


def compute_gradients(image):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    dx = cv2.filter2D(image, -1, kernel_x)
    dy = cv2.filter2D(image, -1, kernel_y)
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) * (180 / np.pi)
    angle[angle < 0] += 360
    return magnitude, angle

def assign_orientation(keypoints, gradients):
    histograms = []
    keypoints_with_orientation=[]
    for kp in keypoints:
        x, y = kp
        hist = np.zeros((36,))
        for i in range(-8, 9):
            for j in range(-8, 9):
                if x + i < 0 or x + i >= gradients.shape[0] or y + j < 0 or y + j >= gradients.shape[1]:
                    continue
                magnitude, angle = gradients[x + i, y + j]
                weight = gaussian_weight(i, j)
                hist[int(angle / 10)] += magnitude * weight
        kp_angle = interpolate_peak(hist)
        histograms.append(hist)
        kp_oriented = kp + (kp_angle,)
        keypoints_with_orientation.append(kp_oriented)
    return histograms,keypoints_with_orientation
def gaussian_weight(x, y, sigma=1.5):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

def interpolate_peak(hist):
    smoothed_hist = np.zeros_like(hist)
    smoothed_hist[1:-1] = (hist[:-2] + hist[1:-1] + hist[2:]) / 3
    smoothed_hist[0] = (hist[0] + hist[-1] + hist[1]) / 3
    smoothed_hist[-1] = (hist[-1] + hist[-2] + hist[0]) / 3
    peaks = np.where((smoothed_hist[:-2] < smoothed_hist[1:-1]) & (smoothed_hist[1:-1] > smoothed_hist[2:]))[0] + 1
    if len(peaks) == 0:
        return 0
    return peaks[np.argmax(hist[peaks])]





# Testing

image = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

magnitude, angle = compute_gradients(image)

gaussian_pyramid = build_scale_space_pyramid(image)
DoG_pyramid = generate_DoG_pyramid(gaussian_pyramid)

# Define threshold for keypoint response
threshold = 100
#
# Detect keypoints
keypoints = detect_keypoints(DoG_pyramid, threshold)
print(len(keypoints))

histograms,bebo = assign_orientation(keypoints, np.dstack((magnitude, angle)))


# Visualize the histogram for the first keypoint
plt.bar(np.arange(0, 360, 10), histograms[0], width=10, align='center')
plt.xlabel('Orientation (degrees)')
plt.ylabel('Magnitude')
plt.title('Gradient Orientation Histogram')
plt.show()


# Visualize keypoints 
# keypoint_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for idx, point in enumerate(keypoints):
#     color = (idx * 10 % 256, idx * 20 % 256, idx * 30 % 256)
#     cv2.circle(keypoint_image, (point[1], point[0]), 3, color, 1) 

# cv2.imshow("Keypoints", keypoint_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
