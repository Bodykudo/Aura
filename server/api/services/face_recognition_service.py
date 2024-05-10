import numpy as np
import os
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC

#TODO :
# 1 - train data with SVM model and remove or leave euclidean distanceself.
# 2 - report Preformance -> confusion matrix.
# 3 - Plot ROC. 
# 4 - combine face detection with face recognition together. 
# 5 - saving data into csv still has some issues.

class FaceRecognition:
    def __init__(self, training_path="./playground/Avengers/train", training_csv="./playground/Avengers/train/pca_data.csv", test_path = None, test_csv=None):
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

    def evaluate_match_svm(self, proj_data, normalised_uface_vector, trainLabels):
        clf = SVC(kernel="linear")
        clf.fit(proj_data.T, trainLabels)
        predicted_label = clf.predict(normalised_uface_vector.reshape(1, -1))
        return predicted_label[0]

    def predict_image_label(self, method, eigenfaces, weights, normalized_test_image, train_imgs_paths, trainLabels):
        if method == "euclidean":
            return self.evaluate_match_euclidean(
                eigenfaces, weights, normalized_test_image, train_imgs_paths, trainLabels
            )
        elif method == "svm":
            return self.evaluate_match_svm(eigenfaces, normalized_test_image, trainLabels)
        else:
            raise ValueError("Invalid evaluation method. Choose 'euclidean' or 'svm'.")


# Example usage
face_recognition = FaceRecognition()
path__Rec_img = "./playground/Avengers/train/chris_hemsworth/chris_hemsworth2.png"
unknown_face = cv2.imread(path__Rec_img)
unknown_face = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(unknown_face, (64, 64))
test_face = np.array(gray, dtype="float64").flatten()
eigenvalues, eigenfaces, weights, labels, image_paths, mean_face, normalised_uface_vector = face_recognition.preprocess_unknown_face(test_face)
reduced_eigenvalues, reduced_eigenvectors, reduced_weights = face_recognition.reduce_eigenfaces(eigenvalues, eigenfaces, weights, threshold=0.9)
train_imgs_paths, trainLabels, best_match = face_recognition.predict_image_label(
    "euclidean", eigenfaces, weights, normalised_uface_vector, image_paths, labels
)

best_match_img = cv2.imread(train_imgs_paths[best_match])
cv2.imshow("training", best_match_img)
cv2.imshow("test", unknown_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

