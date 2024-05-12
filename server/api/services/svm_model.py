import numpy as np
import os
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle


def load_and_apply_PCA(image_path, pca):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    feature_vector = image.flatten()
    transformed_features = pca.transform([feature_vector])

    return transformed_features.flatten()


def train_svm(k):
    features = []
    labels = []

    for person_id in range(1, 5):
        person_dir = rf"C:\Users\hp\Desktop\CV-Tool-Kit\Aura\playground\Avengers\train\{person_id}"
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = img.flatten()
            if img is not None:
                img_normalized = (img - np.mean(img)) / np.std(img)
                features.append(img_normalized)
                labels.append(person_id)

    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pca = PCA(n_components=k, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1e-3, 1e-4],
        "kernel": ["linear", "rbf"],
    }
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid_search.fit(X_train_pca, y_train)

    # Get the best model
    svm_model = grid_search.best_estimator_

    # Test SVM
    y_pred = svm_model.predict(X_test_pca)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return svm_model, pca


def save_model(svm_model, pca, svm_model_path, pca_path):
    # Save SVM model
    with open(svm_model_path, "wb") as f:
        pickle.dump(svm_model, f)

    # Save PCA object
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)


svm_model, pca = train_svm(150)
save_model(svm_model, pca, "svm_model.pkl", "pca.pkl")
