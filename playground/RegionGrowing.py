import cv2
import numpy as np
import random

class Region_Growing():
    def __init__(self, img, threshold, neighboursNum=4):
        self.img = img
        self.h, self.w = img.shape
        self.segmentation = np.zeros(img.shape, dtype=np.uint8)
        self.threshold = threshold
        self.seeds = []
        if neighboursNum == 4:
            self.orientations = [(1,0),(0,1),(-1,0),(0,-1)]
        elif neighboursNum == 8:
            self.orientations = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)] # 8 connectivity

    def set_seeds(self):
        self.seeds = []
        # Selecting 10 seeds randomly
        self.seeds.append((int(self.h/2),int(self.w/2)))
        self.seeds.append((int(2*self.h/3),int(2*self.w/3)))
        self.seeds.append((int(self.h/3-10),int(2*self.w/3)))

    def segment(self):
        for seed in self.seeds:
            curr_pixel = [seed[1], seed[0]]
            # Checking visited pixels
            if self.segmentation[curr_pixel[0], curr_pixel[1]] == 255:
                continue
            contour = []
            seg_size = 1
            # Initialize mean with current pixel
            mean_seg_value = (self.img[curr_pixel[0],curr_pixel[1]])
            dist = 0

            while(dist < self.threshold):
                # Mark as visited
                self.segmentation[curr_pixel[0], curr_pixel[1]] = 255
                # Explore neighbours of current pixel
                contour = self.__explore_neighbours(contour, curr_pixel)
                # Get the nearest neighbour
                nearest_neighbour_idx, dist = self.__get_nearest_neighbour(contour, mean_seg_value)
                # If no more neighbours to grow, move to the next seed
                if nearest_neighbour_idx == -1:
                    break
                # Update current pixel to the nearest neighbour and increment size
                curr_pixel = contour[nearest_neighbour_idx]
                seg_size += 1
                # Update mean pixel value for segmentation
                mean_seg_value = (mean_seg_value * seg_size + float(self.img[curr_pixel[0],curr_pixel[1]])) / (seg_size + 1)
                # Delete from contour once the nearest neighbour is chosen as the current node for expansion
                del contour[nearest_neighbour_idx]

        return self.segmentation

    def display_and_resegment(self, name="Region Growing"):
        # Generate random colors for each segmented region
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(256)]
        colored_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for y in range(self.h):
            for x in range(self.w):
                label = self.segmentation[y, x]
                if label != 0:
                    colored_image[y, x] = colors[label]
                else:
                    colored_image[y, x] = self.img[y, x]

        return colored_image

    def __explore_neighbours(self, contour, current_pixel):
        for orientation in self.orientations:
            neighbour = self.__get_neighbouring_pixel(current_pixel, orientation, self.img.shape)
            if neighbour is None:
                continue
            if self.segmentation[neighbour[0],neighbour[1]] == 0:
                contour.append(neighbour)
                self.segmentation[neighbour[0],neighbour[1]] = 150
        return contour

    def __get_neighbouring_pixel(self, current_pixel, orient, img_shape):
        # Getting neighbour
        neighbour = ((current_pixel[0] + orient[0]), (current_pixel[1] + orient[1]))
        # Checking if pixel is out of image boundaries
        if self.is_pixel_inside_image(pixel=neighbour, img_shape=img_shape):
            return neighbour
        else:
            return None

    def __get_nearest_neighbour(self, contour, mean_seg_value):
        dist_list = [abs(int(self.img[pixel[0], pixel[1]]) - mean_seg_value) for pixel in contour]
        if len(dist_list) == 0:
            return -1, 1000
        min_dist = min(dist_list)
        index = dist_list.index(min_dist)
        return index, min_dist

    def is_pixel_inside_image(self, pixel, img_shape):
        return 0 <= pixel[0] < img_shape[0] and 0 <= pixel[1] < img_shape[1]

# Helper functions
def resize_image(image_data):
    if image_data.shape[0] > 1000:
        image_data = cv2.resize(image_data, (0,0), fx=0.25, fy=0.25)
    if image_data.shape[0] > 500:
        image_data = cv2.resize(image_data, (0,0), fx=0.5, fy=0.5)
    return image_data

def apply_gaussian_smoothing(image_data, filter_size=3):
    return cv2.GaussianBlur(image_data, (filter_size,filter_size), 0)

# Main function to run region growing segmentation
def run_region_growing(image_path, threshold, neighboursNum):
    image_data = cv2.imread(image_path, 0)
    image_data = resize_image(image_data)
    image_data_post_smoothing = apply_gaussian_smoothing(image_data)
    region_growing_instance = Region_Growing(image_data_post_smoothing, threshold=threshold, neighboursNum=neighboursNum)
    region_growing_instance.set_seeds()
    result = region_growing_instance.segment()
    return result

# Define the image path
image_path = "C:/College/3rd Year/Second Term/Computer Vision/Aura/playground/coin.jpg"

# Define the threshold and number of neighbors
threshold = 20
neighboursNum = 4

# Run region growing segmentation
segmented_image = run_region_growing(image_path, threshold, neighboursNum)
colored_image = 

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
