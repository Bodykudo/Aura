import numpy as np
import os
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
import pickle


class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        # Compute mean along columns (features)
        self.mean_ = np.mean(X, axis=0)

        # Center the data
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvectors and eigenvalues of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors based on eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, idx[: self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FaceRecognition:
    def __init__(
        self,
        training_path="./playground/Avengers/train",
        training_csv="./playground/Avengers/train/pca_data.csv",
        test_path=None,
        test_csv=None,
    ):
        self.training_path = training_path
        self.training_csv = training_csv
        self.test_path = test_path
        self.test_csv = test_csv
        self.mean_face = None

    ### --------------------------------- EUCLIDEAN DISTANCE ---------------------------------------------########
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
        num_components = 3
        reduced_eigenvalues = eigenvalues[:num_components]
        reduced_eigenfaces = eigenfaces[:num_components]
        reduced_weights = weights[:, :num_components]
        return reduced_eigenvalues, reduced_eigenfaces, reduced_weights

    def apply_face_recognition(self, unknown_face, test_labels):
        if os.path.isfile(self.training_csv):
            pca_df = pd.read_csv(self.training_csv)
            eigenvalues = np.array(pca_df["Eigenvalues"])
            eigenfaces = np.array(pca_df["Eigenfaces"])
            weights = np.array(pca_df["Weights"])
            labels = np.array(pca_df["Labels"])
            image_paths = np.array(pca_df["ImagePaths"])
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
        print(reduced_eigenfaces.shape)
        # self.save_pca_data_to_csv(eigenvalues, eigenfaces, weights, labels, image_paths)
        labels = train_data_labels
        # svm_model = self.train_svm(reduced_eigenfaces, reduced_weights, labels)
        # _, centred_test_faces = self.get_mean_face_normalized(
        #         images=unknown_faces
        #     )
        # eigenvalues, eigenfaces, weights = self.pca(centred_test_faces)
        # self.predict_svm(svm_model, eigenfaces, test_labels)
        best_matches = []
        # Perform face recognition for each unknown face

        normalized_unknown_face_vector = unknown_face.flatten() - self.mean_face
        result = self.predict_image_label(
            "euclidean",
            reduced_eigenfaces,
            reduced_weights,
            normalized_unknown_face_vector,
        )
        best_matches.append(result)
        # matched_labels = [labels[index] for index in best_matches]
        return train_data_paths, labels, best_matches

    def evaluate_match_euclidean(self, eigen_faces, w, normalised_unknown_face_vector):
        w_unknown = np.dot(eigen_faces, normalised_unknown_face_vector)
        euclidean_distance = np.linalg.norm(w - w_unknown, axis=1)
        best_match = np.argmin(euclidean_distance)
        return best_match

    def evaluate_match_svm(self, eigen_faces, unknown_face_vector, train_labels):

        clf = SVC(kernel="linear")
        clf.fit(eigen_faces.T, train_labels)
        predicted_label = clf.predict(unknown_face_vector.reshape(1, -1))
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

    def save_pca_data_to_csv(
        self, eigenvalues, eigenfaces, weights, labels, image_paths
    ):
        eigenvalues = np.array(eigenvalues)
        eigenfaces = np.array(eigenfaces)
        weights = np.array(weights)
        data = {
            "Eigenvalues": eigenvalues.tolist(),
            "Eigenfaces": eigenfaces.tolist(),
            "Weights": weights.tolist(),
            "Labels": labels,
            "ImagePaths": image_paths,
        }
        pca_df = pd.DataFrame(data)
        pca_df.to_csv(self.training_csv, index=False)

    ### --------------------------------- EUCLIDEAN DISTANCE ---------------------------------------------########

    ### --------------------------------- SVM MODEL ---------------------------------------------########
    def load_model(self, svm_model_path, pca_path):
        # Load SVM model
        with open(svm_model_path, "rb") as f:
            svm_model = pickle.load(f)

        # Load PCA object
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        return svm_model, pca

    def predict_face(self, img, pca, svm_model):
        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))  # Keep the size consistent with training
        img = img.flatten()

        # Normalize the image
        img_normalized = (img - np.mean(img)) / np.std(img)

        img_pca = pca.transform([img_normalized])
        prediction = svm_model.predict(img_pca)
        probability = svm_model.predict_proba(img_pca)
        mapping = ["chris evans", "chris hemsworth", "mark ruffalo", "robert downey jr"]

        # Thresholding
        max_prob = max(max(probability))
        if max_prob > 0.5:
            decision = prediction[0]
            decision = mapping[decision - 1]
        else:
            decision = "Not Recognized"
        return decision

    def load_and_apply_PCA(image_path, pca):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        feature_vector = image.flatten()
        transformed_features = pca.transform([feature_vector])

        return transformed_features.flatten()
