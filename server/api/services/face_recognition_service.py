import numpy as np
import cv2
import pickle

from api.services.face_detection_service import FaceDetection
from api.utils import read_image


class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
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
        cascade_file: str,
        svm_file: str,
        pca_file: str,
    ):
        self.cascade_file = cascade_file
        self.svm_model, self.pca = self.load_model(svm_file, pca_file)

    def load_model(self, svm_model_path, pca_path):
        with open(svm_model_path, "rb") as f:
            svm_model = pickle.load(f)

        # Load PCA object
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        return svm_model, pca

    def predict_face(
        self,
        image,
    ):
        image = cv2.resize(image, (64, 64))
        image = image.flatten()

        image_normalized = (image - np.mean(image)) / np.std(image)

        image_transformed = self.pca.transform([image_normalized])
        prediction = self.svm_model.predict(image_transformed)
        probability = self.svm_model.predict_proba(image_transformed)
        mapping = ["Captain America", "Thor", "Hulk", "Iron Man"]

        max_prob = max(max(probability))
        if max_prob > 0.5:
            decision = prediction[0]
            decision = mapping[decision - 1]
        else:
            decision = "Not Recognized"
        return decision

    def recognize_faces(self, image_path: str):
        face_detection = FaceDetection(self.cascade_file)
        image = read_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for _, (x, y, w, h) in enumerate(face_detection.detect_faces(image), start=0):
            face_region = gray[y : y + h, x : x + w]
            predicted = self.predict_face(face_region)
            color = (0, 255, 0) if predicted != "Not Recognized" else (255, 0, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                predicted,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return image
