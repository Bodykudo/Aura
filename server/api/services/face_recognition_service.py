import numpy as np
import os
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC


class FaceRecognition:
    def __init__(
        self,
        training_path="./Avengers/train",
        training_csv="./playground/Avengers/train/pca_data.csv",
        test_path=None,
        test_csv=None,
    ):
        self.training_path = training_path
        self.training_csv = training_csv
        self.test_path = test_path
        self.test_csv = test_csv
        self.mean_face = None

    def get_image_paths(self, path):
        image_paths = []
        labels = []
        for subdir in os.listdir(path):
            label = subdir
            subdir_path = os.path.join(path, subdir)
            for filename in os.listdir(subdir_path):
                labels.append(label)
                image_path = os.path.join(subdir_path, filename)
                image_paths.append(image_path)
        return image_paths, labels

    def get_array_images(self, image_paths, width=64, height=64):
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (width, height))
            images.append(image.flatten())
        return np.array(images)

    def get_mean_face_normalized(self, images):
        mean_face = np.mean(images, axis=0)
        centred_faces = images - mean_face
        return mean_face, centred_faces

    def pca(self, normalized_faces):
        cov_matrix = np.cov(normalized_faces)
        cov_matrix = np.divide(cov_matrix, float(len(normalized_faces)))
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[idx]
        eigenvectors_sorted = eigenvectors[:, idx]
        eigenvectors_sorted = preprocessing.normalize(eigenvectors_sorted)
        eigenvectors_sorted = np.dot(
            normalized_faces.transpose(), eigenvectors_sorted.transpose()
        )
        eigenvectors_sorted = eigenvectors_sorted.transpose()
        weights = np.array(
            [np.dot(eigenvectors_sorted, face) for face in normalized_faces]
        )
        return eigenvalues_sorted, eigenvectors_sorted, weights

    def reduce_eigenfaces(self, eigenvalues, eigenfaces, weights, threshold=0.9):
        var_comp_sum = np.cumsum(eigenvalues) / sum(eigenvalues)
        num_components = np.where(var_comp_sum < threshold)[0][-1] + 1
        reduced_eigenvalues = eigenvalues[:num_components]
        reduced_eigenfaces = eigenfaces[:num_components]
        reduced_weights = weights[:, :num_components]
        return reduced_eigenvalues, reduced_eigenfaces, reduced_weights

    def apply_face_recognition(self, unknown_faces):
        if os.path.isfile(self.training_csv):
            pca_df = pd.read_csv(self.training_csv)
            eigenvalues = np.array(pca_df['Eigenvalues'])
            eigenfaces = np.array(pca_df['Eigenfaces'])
            weights = np.array(pca_df['Weights'])
            labels = np.array(pca_df['Labels'])
            
        else:
            train_data_paths, train_data_labels = self.get_image_paths(
                path=self.training_path
            )
            training_images = self.get_array_images(train_data_paths)
            self.mean_face, centred_training_faces = self.get_mean_face_normalized(
                images=training_images
            )
            eigenvalues, eigenfaces, weights = self.pca(centred_training_faces)
            
            
        reduced_eigenvalues, reduced_eigenfaces, reduced_weights = (
            self.reduce_eigenfaces(eigenvalues, eigenfaces, weights)
        )
        # self.save_pca_data_to_csv(eigenvalues, eigenfaces, weights, labels, image_paths)
        labels = train_data_labels

        best_matches = []
        # Perform face recognition for each unknown face
        for unknown_face in unknown_faces:
            normalized_unknown_face_vector = unknown_face.flatten() - self.mean_face
            result = self.predict_image_label(
                "euclidean",
                reduced_eigenfaces,
                reduced_weights,
                normalized_unknown_face_vector,
            )
            best_matches.append(result)
        # matched_labels = [labels[index] for index in best_matches]
        return labels, best_matches

    def evaluate_match_euclidean(self, eigen_faces, w, normalised_uface_vector):
        w_unknown = np.dot(eigen_faces, normalised_uface_vector)
        euclidean_distance = np.linalg.norm(w - w_unknown, axis=1)
        best_match = np.argmin(euclidean_distance)
        return best_match

    def evaluate_match_svm(self, eigen_faces, normalised_uface_vector, trainLabels):
        clf = SVC(kernel="linear")
        clf.fit(eigen_faces.T, trainLabels)
        predicted_label = clf.predict(normalised_uface_vector.reshape(1, -1))
        return predicted_label[0]

    def predict_image_label(self, method, eigenfaces, weights, normalized_test_image):
        if method == "euclidean":
            return self.evaluate_match_euclidean(
                eigenfaces, weights, normalized_test_image
            )
        elif method == "svm":
            return self.evaluate_match_svm(eigenfaces, normalized_test_image)
        else:
            raise ValueError("Invalid evaluation method. Choose 'euclidean' or 'svm'.")

    def save_pca_data_to_csv(self, eigenvalues, eigenfaces, weights, labels):
        eigenvalues = np.array(eigenvalues)
        eigenfaces = np.array(eigenfaces)
        weights = np.array(weights)
        data = {
            "Eigenvalues": eigenvalues.tolist(),
            "Eigenfaces": eigenfaces.tolist(),
            "Weights": weights.tolist(),
            "Labels": labels,
        }
        pca_df = pd.DataFrame(data)
        pca_df.to_csv(self.training_csv, index=False)





# Example usage
face_recognition = FaceRecognition()
test_faces_folder = "./Avengers/test"
test_images_paths = [
    os.path.join(test_faces_folder, subdir, filename)
    for subdir in os.listdir(test_faces_folder)
    for filename in os.listdir(os.path.join(test_faces_folder, subdir))
]

faces = []
test_labels = []
for test_image_path in test_images_paths:
    unknown_face = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    unknown_face = cv2.resize(unknown_face, (64, 64))
    unknown_face = np.array(unknown_face, dtype="float64")
    faces.append(unknown_face)
    test_labels.append(
        os.path.basename(os.path.dirname(test_image_path))
    )  # Extract test label from the directory name


labels, best_matches = face_recognition.apply_face_recognition(faces)
matched_labels = [
    labels[index] for index in best_matches
]  # Get the label of the matched image

for i in range(len(test_labels)):
    print(
        i, "-", " Test Label:", test_labels[i], "=", "Matched Label:", matched_labels[i]
    )
