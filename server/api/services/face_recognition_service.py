import numpy as np
import os
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


#TODO :
# 1 - train data with SVM model and remove or leave euclidean distanceself.
# 2 - report Preformance -> confusion matrix.
# 3 - Plot ROC. 
# 4 - combine face detection with face recognition together. 
# 5 - saving data into csv still has some issues.

class FaceRecognition:
    def __init__(self, training_path=r"C:\Users\hp\Desktop\CV-Tool-Kit\Aura\playground\Avengers\train", training_csv=r"C:\Users\hp\Desktop\CV-Tool-Kit\Aura\playground\Avengers\train\pca_data.csv", test_path = None, test_csv=None):
        self.training_path = training_path
        self.training_csv = training_csv
        self.test_path = test_path
        self.test_csv = test_csv
        
        
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

    def pca(self, train_images, normalized_faces):
        cov_matrix = np.cov(normalized_faces)
        cov_matrix = np.divide(cov_matrix, float(len(normalized_faces)))
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
        eig_pairs.sort(reverse=True)
        eigenvalues_sorted = [eig_pairs[index][0] for index in range(len(eigenvalues))]
        eigenvectors_sorted = [eig_pairs[index][1] for index in range(len(eigenvalues))]
        eigenvectors_sorted = preprocessing.normalize(eigenvectors_sorted)
        eigenvectors_sorted = np.dot(train_images.transpose(), eigenvectors_sorted.transpose())
        eigenvectors_sorted = eigenvectors_sorted.transpose()
        weights = np.array([np.dot(eigenvectors_sorted, face) for face in normalized_faces])
        return eigenvalues_sorted, eigenvectors_sorted, weights

    def save_pca_data_to_csv(self, eigenvalues, eigenfaces, weights, labels, image_paths):
        eigenvalues = np.array(eigenvalues)
        eigenfaces = np.array(eigenfaces)
        weights = np.array(weights)
        data = {'Eigenvalues': eigenvalues.tolist(),
                'Eigenfaces': eigenfaces.tolist(),
                'Weights': weights.tolist(),
                'Labels': labels,
                'ImagePaths': image_paths}
        pca_df = pd.DataFrame(data)
        pca_df.to_csv(self.training_csv, index=False)

    def preprocess_unknown_face(self, unknown_face):
        if os.path.isfile(self.training_csv):
            pca_df = pd.read_csv(self.training_csv)
            eigenvalues = np.array(pca_df['Eigenvalues'])
            eigenfaces = np.array(pca_df['Eigenfaces'])
            weights = np.array(pca_df['Weights'])
            labels = np.array(pca_df['Labels'])
            image_paths = np.array(pca_df['ImagePaths'])
            mean_face = np.mean(eigenfaces.astype(np.float64), axis=0) 
            normalised_uface_vector = unknown_face.flatten() - mean_face
        else:
            train_imgs_paths, trainLabels = self.get_image_paths(path=self.training_path)
            training_images = self.get_array_images(train_imgs_paths)
            mean_face, centred_training_faces = self.get_mean_face_normalized(images=training_images)
            eigenvalues, eigenfaces, weights = self.pca(training_images, centred_training_faces)
            labels = trainLabels
            image_paths = train_imgs_paths
            # self.save_pca_data_to_csv(eigenvalues, eigenfaces, weights, labels, image_paths)
            normalised_uface_vector = unknown_face.flatten() - mean_face
        return eigenvalues, eigenfaces, weights, labels, image_paths, mean_face, normalised_uface_vector

    def reduce_eigenfaces(self, eigenvalues, eigenfaces, weights, threshold=0.9):
        var_comp_sum = np.cumsum(eigenvalues) / sum(eigenvalues)
        num_components = np.where(var_comp_sum < threshold)[0][-1] + 1
        reduced_eigenvalues = eigenvalues[:num_components]
        reduced_eigenfaces = eigenfaces[:num_components]
        reduced_weights = weights[:, :num_components]
        return reduced_eigenvalues, reduced_eigenfaces, reduced_weights

    def evaluate_match_euclidean(self, proj_data, w, normalised_uface_vector, train_imgs_paths, trainLabels):
        w_unknown = np.dot(proj_data, normalised_uface_vector)
        euclidean_distance = np.linalg.norm(w - w_unknown, axis=1)
        best_match = np.argmin(euclidean_distance)
        return train_imgs_paths, trainLabels, best_match

    def load_and_apply_PCA(self,image_path, pca):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        feature_vector = image.flatten()
        transformed_features = pca.transform([feature_vector])

        return transformed_features.flatten()
    def evaluate_match_svm(self,k):
        features = []
        labels = []

        for person_id in range(1, 5):
            person_dir = rf"C:\Users\hp\Desktop\CV-Tool-Kit\Aura\playground\Avengers\train\{person_id}"
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                img= img.flatten()
                if img is not None:
                        img_normalized = (img - np.mean(img)) / np.std(img)
                        features.append(img_normalized)
                        labels.append(person_id)

        X = np.array(features)
        y=np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pca = PCA(n_components=k, whiten=True, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1e-3, 1e-4],
                      'kernel': ['linear', 'rbf']}
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
        grid_search.fit(X_train_pca, y_train)

        # Get the best model
        svm_model = grid_search.best_estimator_

        # Test SVM
        y_pred = svm_model.predict(X_test_pca)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        return svm_model,pca

    def predict_face(self, image_path, pca, svm_model):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))  # Keep the size consistent with training
        img = img.flatten()

        # Normalize the image
        img_normalized = (img - np.mean(img)) / np.std(img)

        img_pca = pca.transform([img_normalized])
        prediction = svm_model.predict(img_pca)
        probability = svm_model.predict_proba(img_pca)
        mapping = ['chris evans', 'chris hemsworth', 'mark ruffalo', 'robert downey jr']

        # Thresholding
        max_prob = max(max(probability))
        if max_prob > 0.5:
            decision = prediction[0]
            decision = mapping[decision - 1]
        else:
            decision = "Not Recognized"
        return decision

    def predict_image_label(self, method,image_path, eigenfaces, weights, normalized_test_image, train_imgs_paths, trainLabels):
        if method == "euclidean":
            return self.evaluate_match_euclidean(
                eigenfaces, weights, normalized_test_image, train_imgs_paths, trainLabels
            )
        elif method == "svm":
           svm_model,pca=self.evaluate_match_svm(150)
           final_decision=self.predict_face(image_path,pca,svm_model)
           return None,final_decision,None
        else:
            raise ValueError("Invalid evaluation method. Choose 'euclidean' or 'svm'.")


# Example usage
face_recognition = FaceRecognition()
path__Rec_img = r"C:\Users\hp\Downloads\mark-ruffalo_g7rr.jpg"
unknown_face = cv2.imread(path__Rec_img,cv2.IMREAD_GRAYSCALE)
# unknown_face = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(unknown_face, (64, 64))
test_face = np.array(gray, dtype="float64").flatten()
eigenvalues, eigenfaces, weights, labels, image_paths, mean_face, normalised_uface_vector = face_recognition.preprocess_unknown_face(test_face)
reduced_eigenvalues, reduced_eigenvectors, reduced_weights = face_recognition.reduce_eigenfaces(eigenvalues, eigenfaces, weights, threshold=0.9)
train_imgs_paths, trainLabels, best_match = face_recognition.predict_image_label(
    "svm",path__Rec_img, eigenfaces, weights, normalised_uface_vector, image_paths, labels
)

print(trainLabels)

# best_match_img = cv2.imread(train_imgs_paths[best_match])
# cv2.imshow("training", best_match_img)
# cv2.imshow("test", unknown_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

