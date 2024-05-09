import os
import cv2
import numpy as np

def pca_from_scratch(face_data):
    # Step 1: Construct data matrix in shape of (No. of pixels * No. of images)
    data_matrix = np.array([image.flatten() for folder_images in face_data for image in folder_images]).T

    # Step 2: Get the Mean Image (Average Column)
    mean_face_image = data_matrix.mean(axis=1, keepdims=True)

    # Step 3: Subtract the mean image from All images
    centered_data = data_matrix - mean_face_image[:, np.newaxis]

    # Step 4: Get the Covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Step 5: Compute eigenvalues and eigenvectors using a more efficient method
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    # Step 6: Normalize the eigenvectors
    normalized_eigenvectors = eigen_vectors / np.linalg.norm(eigen_vectors, axis=1)[:, np.newaxis]

    # Step 7: Keep all vectors summing up eigenvalues to 90% and remove the rest
    eigen_value_sum = np.sum(eigen_values)
    eigen_value_cumsum = np.cumsum(eigen_values)
    cutoff_index = np.argmax(eigen_value_cumsum >= 0.9 * eigen_value_sum) + 1
    selected_eigenvectors = normalized_eigenvectors[:, :cutoff_index]

    # Step 8: Map all images to new components
    images_weights = np.dot(selected_eigenvectors.T, centered_data)

    return images_weights, selected_eigenvectors, mean_face_image



def import_face_data(dataset_folder,  resize_shape=(100, 100)):
    face_data = []
    labels = {}
    label_count = 0

    # Check if the dataset folder exists
    if not os.path.isdir(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' not found.")
        return None, None

    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_images = []
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, resize_shape)
                    folder_images.append(img_resized)
                else:
                    print(f"Error: Unable to read image '{image_path}'.")
            face_data.append(folder_images)
            labels[label_count] = folder_name
            label_count += 1

    return face_data, labels

# Example usage:
dataset_folder = "./playground/Avengers/train"
face_data, labels = import_face_data(dataset_folder)

for folder_index, folder_images in enumerate(face_data):
    print(f"Folder {labels[folder_index]}:")
    for image_index, image in enumerate(folder_images):
        print(f"\tImage {image_index}: {image.shape}")

# Print the labels
print("\nLabels:")
for label_index, label_name in labels.items():
    print(f"{label_index}: {label_name}")


# Example usage:
mapped_images, selected_eigenvectors, mean_image = pca_from_scratch(face_data)

# Print the shapes of mapped images, selected eigenvectors, and mean image
print("Mapped Images Shape:", mapped_images.shape)
print("Selected Eigenvectors Shape:", selected_eigenvectors.shape)
print("Mean Image Shape:", mean_image.shape)