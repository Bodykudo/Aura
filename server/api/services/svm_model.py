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
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


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
        self.components_ = eigenvectors[:, idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
   
   
def load_and_apply_PCA(image_path, pca):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    feature_vector = image.flatten()
    transformed_features = pca.transform([feature_vector])

    return transformed_features.flatten() 

def calculate_metrics_per_class(y_test, y_pred):
    for i in range(len(y_test)):
        True_predctions = np.sum((y_test == y_pred))  # True Positives
        false_predictions = np.sum((y_test != y_pred))  # False Positives
    return True_predctions, false_predictions

def train_svm(k):
    features = []
    labels = []


    for person_id in range(1, 6):
        person_dir = rf"./playground/Avengers/train/{person_id}"
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

    # pca = PCA(n_components=k, whiten=True, random_state=42)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    
    pca = MyPCA(n_components=k)
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

def predict_face(image_path, pca, svm_model):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    
def load_model(svm_model_path, pca_path):
        # Load SVM model
        with open(svm_model_path, "rb") as f:
            svm_model = pickle.load(f)

        # Load PCA object
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        return svm_model, pca  
    

svm_model, pca = train_svm(150)

save_model(svm_model, pca, "svm_model.pkl", "pca.pkl")
# svm_model, pca = load_model("svm_model.pkl", "pca.pkl")

ds = predict_face("./playground/Avengers/test/markk.jpeg", pca, svm_model)
print(ds)


"""

    PERFORMANCE EVALUATION
    

    # Calculate True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
    trues, falses  = calculate_metrics_per_class(y_test, y_pred)
    # Calculate False Positive Rate (FPR), True Positive Rate (TPR), and Selection Error (SLE)
    print(trues)
    print(falses)
    
    y_test_binary = label_binarize(y_test, classes=[0, 1, 2, 3, 4])  # Update the classes as per your data
    y_pred_binary = label_binarize(y_pred, classes=[0, 1, 2, 3, 4])
    
    conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_pred_binary[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure()
    for i in range(5):
        plt.plot(fpr[i], tpr[i], label=f'Class {i + 1} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
"""