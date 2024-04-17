import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key
def SIFT(image):
    # Define threshold for keypoint response
    threshold = 100
    gaussian_pyramid = build_scale_space_pyramid(image)
    DoG_pyramid = generate_DoG_pyramid(gaussian_pyramid)
    magnitude, angle = compute_gradients(image)
    # Detect keypoints
    keypoints = detect_keypoints(DoG_pyramid, threshold)
    print(len(keypoints))
    print(keypoints)
    histograms, keypoints = assign_orientation(keypoints, np.dstack((magnitude, angle)))
    descriptors = keypoint_descriptor(image, keypoints)
    print(len(descriptors))
    return keypoints, descriptors
        
    
    
    
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

def detect_keypoints(DoG_pyramid, threshold, edge_threshold=0.03):
    keypoints = []
    for octave, octave_images in enumerate(DoG_pyramid):
        for scale, img in enumerate(octave_images[1:-1]):  
            prev_img = octave_images[scale]
            next_img = octave_images[scale + 2]
            keypoints.extend(detect_keypoints_in_image(img, prev_img, next_img, threshold, edge_threshold, 10))
    return keypoints

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
                    # Append keypoint as a tuple (x, y, size, octave)
                    keypoints.append((i, j))  # Default size and octave for now
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
    
def compareKeypoints(keypoint1, keypoint2):
    """Custom comparison function for keypoints."""
    if keypoint1[0] != keypoint2[0]:
        return keypoint1[0] - keypoint2[0]
    if keypoint1[1] != keypoint2[1]:
        return keypoint1[1] - keypoint2[1]
    if keypoint1[2] != keypoint2[2]:
        return keypoint2[2] - keypoint1[2]
    if keypoint1[3] != keypoint2[3]:
        return keypoint1[3] - keypoint2[3]
    if keypoint1[4] != keypoint2[4]:
        return keypoint2[4] - keypoint1[4]
    if keypoint1[5] != keypoint2[5]:
        return keypoint2[5] - keypoint1[5]
    if keypoint1[6] != keypoint2[6]:
        return keypoint2[6] - keypoint1[6]
    return 0  # Keypoints are equal

# Modify the removeDuplicateKeypoints function accordingly
def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints."""
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint[0] != next_keypoint[0] or \
           last_unique_keypoint[1] != next_keypoint[1] or \
           last_unique_keypoint[2] != next_keypoint[2] or \
           last_unique_keypoint[3] != next_keypoint[3]:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

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
    return histograms, keypoints_with_orientation

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

def keypoint_descriptor(image, keypoints):
    descriptors = []

def keypoint_descriptor(image, keypoints):
    descriptors = []

    for kp in keypoints:
        descriptor = compute_descriptor(image, kp[0], kp[1])
        descriptors.append(descriptor)

    return np.array(descriptors)



def compute_descriptor(image, x, y, block_size=4, num_bins=8):
    descriptor = []

    # Extract 16x16 neighborhood around the keypoint
    patch = image[max(y - 8, 0):min(y + 8, image.shape[0]),
                  max(x - 8, 0):min(x + 8, image.shape[1])]

    # Divide the patch into 16 sub-blocks of 4x4 size
    sub_blocks = [patch[i:i + block_size, j:j + block_size]
                  for i in range(0, patch.shape[0], block_size)
                  for j in range(0, patch.shape[1], block_size)]

    for sub_block in sub_blocks:
        # Compute gradient magnitude and orientation
        gradient_magnitude, gradient_orientation = compute_gradient(sub_block)

        # Compute histogram of orientations with 8 bins
        histogram = np.zeros(num_bins, dtype=np.float32)
        bin_width = 360 / num_bins
        for row in range(sub_block.shape[0]):
            for col in range(sub_block.shape[1]):
                angle = gradient_orientation[row, col] % 360
                bin_index = int(angle / bin_width)
                histogram[bin_index] += gradient_magnitude[row, col]

        # Concatenate the histogram to the descriptor
        descriptor.extend(histogram)

    # Normalize the descriptor
    descriptor /= np.linalg.norm(descriptor)

    # Clip the descriptor values to ensure they are within the range [0, 0.2]
    np.clip(descriptor, 0, 0.2, out=descriptor)

    # Rescale the descriptor to an integer value between 0 and 255
    descriptor *= 255
    descriptor = np.clip(descriptor, 0, 255).astype(np.uint8)

    return descriptor

def compute_gradient(patch):
    # Compute gradient in x and y directions using Sobel filters
    gradient_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and orientation
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    return gradient_magnitude, gradient_orientation


# Testing
image1 = cv2.imread("./playground/box.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./playground/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

kp1, ds1 = SIFT(image1)
kp2, ds2 = SIFT(image2)

# ds1_array = np.array(ds1)
# ds2_array = np.array(ds2)


# bf = cv2.BFMatcher()

# # Match descriptors between the images
# matches = bf.knnMatch(ds1, ds2, k=2)

# # Apply ratio test
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append([m])

# # Draw matches
# matched_image = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Display the matched image
# cv2.imshow("Matched Image", matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# Visualize the histogram for the first keypoint
# plt.bar(np.arange(0, 360, 10), histograms[0], width=10, align='center')
# plt.xlabel('Orientation (degrees)')
# plt.ylabel('Magnitude')
# plt.title('Gradient Orientation Histogram')
# plt.show()


# Visualize keypoints 
# keypoint_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for idx, point in enumerate(keypoints):
#     color = (idx * 10 % 256, idx * 20 % 256, idx * 30 % 256)
#     cv2.circle(keypoint_image, (point[1], point[0]), 3, color, 1) 

# cv2.imshow("Keypoints", keypoint_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Testing
# image_path = 'cat.png'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# gaussian_pyramid = build_scale_space_pyramid(image)

# # Generate the Difference-of-Gaussian (DoG) pyramid
# DoG_pyramid = generate_DoG_pyramid(gaussian_pyramid)

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

