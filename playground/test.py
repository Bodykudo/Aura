from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import cv2
from sklearn import preprocessing
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from os import listdir


def get_image_paths(path):

    image_paths = []
    labels = []

    # Loop through each subfolder in the folder
    for subdir in os.listdir(path):
        # Get the label/class from the subfolder name
        label = subdir
        subdir_path = os.path.join(path, subdir)

        # Loop through each image in the subfolder
        for filename in os.listdir(subdir_path):
            # Assign the label to the image
            labels.append(label)
            image_path = os.path.join(subdir_path, filename)
            image_paths.append(image_path)

    return image_paths, labels


def get_array_images(image_paths,width = 64, height = 64):
    images = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height)) # resize the image to a smaller size
        images.append(image.flatten()) # flatten the image into a 1D array
    
    images = np.array(images)

    return images


def get_mean_normalized(images, image_paths, height=64, width=64):

    mean_face = np.array(np.zeros((height * width)))  # Initialize mean face without the extra dimension

    for i in images:
        # print(i.shape)
        mean_face += i  # sum all images

    mean_face /= len(image_paths)  # to get the mean face by dividing the summation by the length of images
    
    # Normalize Faces
    normalized_faces = np.zeros((len(image_paths), height * width))

    for idx, img in enumerate(images):
        normalized_faces[idx] = img - mean_face  # subtract mean face from each image

    return mean_face, normalized_faces


def get_cov_matrix(normalized_faces):

    cov_matrix = cov_matrix = np.cov(normalized_faces)
    cov_matrix = np.divide(cov_matrix, float(len(normalized_faces)))
    return cov_matrix


def get_eigVec_eigVal(cov_matrix):
    (
        eigenvalues,
        eigenvectors,
    ) = np.linalg.eig(cov_matrix)

    eig_pairs = [
        (eigenvalues[index], eigenvectors[:, index])
        for index in range(len(eigenvalues))
    ]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
    eigVec_norm = preprocessing.normalize(eigvectors_sort)  # normalize eigen vectors

    return eigvalues_sort, eigVec_norm


# to get the eigen faces till 90%
def get_reduced_eigvectors(eigvectors, eigvalues_sort):

    var_comp_sum = np.cumsum(eigvalues_sort) / sum(eigvalues_sort)

    reduced_data = []
    for i in var_comp_sum:
        if i < 0.91:
            reduced_data.append(i)

    print(len(var_comp_sum))
    print(len(reduced_data))

    reduced_data = np.array(eigvectors[: len(reduced_data)]).transpose()

    return reduced_data


def get_eigenfaces(train_images, reduced_eigenvectors):
    proj_data = np.dot(train_images.transpose(), reduced_eigenvectors)
    proj_data = proj_data.transpose()
    return proj_data


# get weights of images
def get_weights(eigenfaces, normalized_faces):
    weights = np.array([np.dot(eigenfaces, i) for i in normalized_faces])
    return weights


# apply pca
def PCA_APPLY(unknown_face):
    training_path = "./playground/Avengers/train"
    unknown_face_vector = unknown_face  # get the flattened array
    train_imgs_paths, trainLabels = get_image_paths(path=training_path)
    training_images = get_array_images(train_imgs_paths)
    mean_face, normalised_training = get_mean_normalized(
        images=training_images, image_paths=train_imgs_paths
    )
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)
    cov_matrix = get_cov_matrix(normalised_training)
    eigvalues_sort, eigenfaces = get_eigVec_eigVal(cov_matrix)
    reduced_data = get_reduced_eigvectors(eigenfaces, eigvalues_sort)
    proj_data = get_eigenfaces(training_images, reduced_data)
    w = get_weights(proj_data, normalised_training)
    w_unknown = np.dot(
        proj_data, normalised_uface_vector
    )  # get the weight of test face
    print(np.shape(normalised_uface_vector), np.shape(w_unknown))
    euclidean_distance = np.linalg.norm(
        w - w_unknown, axis=1
    )  # get the euclidean distance
    best_match = np.argmin(euclidean_distance)  # get the index of the best matched one

    print(best_match)
    print(trainLabels[best_match])

    # output_image = training_images[best_match].reshape(64, 64)
    # saved = mpimg.imsave("../images/FaceRecognized.png", output_image, cmap="gray")

    return train_imgs_paths, trainLabels, best_match


def calculate_performance(test_path, threshold=200, width=64, height=64):

    test_image_paths, test_labels = get_image_paths(path=test_path)
    test_images = get_array_images(image_paths=test_image_paths)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tpr_values = []
    fpr_values = []
    bestMatches = []

    for i, img in enumerate(test_images, start=0):
        train_imgs_paths, train_labels, best_match = PCA_APPLY(img)
        bestMatches.append(best_match)
        positive = test_labels[i] == train_labels[best_match]

        if best_match <= threshold:
            if positive == 1:
                print("Matched:" + train_labels[best_match], end="\t")
                tp += 1
            elif positive == 0:
                print("F/Matched:" + train_labels[best_match], end="\t")
                fp += 1

        elif best_match >= threshold:
            if positive == 1:
                print("Unknown face!" + train_labels[best_match], end="\t")
                fn += 1
            elif positive == 0:
                tn += 1
        print(tp, tn, fp, fn)

        if (tp + fn) != 0:
            tpr = tp / (tp + fn)
        elif (tp + fn) == 0:
            tpr = 0

        if (fp + tn) != 0:
            fpr = fp / (fp + tn)
        elif (fp + tn) == 0:
            fpr = 0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    num_images = tp + tn + fp + fn

    FMR = fp / num_images
    FNMR = fn / num_images
    accuracy = (tp + tn) / num_images
    precision = tp / (tp + fp)
    specificity = tn / (tn + fn)

    print(bestMatches)
    print("fpr = {} \t".format(FMR), end=" ")
    print("accuracy = {} \t".format(accuracy), end=" ")
    print("precision = {} \t".format(precision), end=" ")
    print("specificity = {} \t".format(specificity))

    return fpr_values, tpr_values, accuracy, precision, specificity


def ROC_plot(tpr_values, fpr_values):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values)
    plt.plot([0, 1], [0, 1], "--", color="gray")  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    # plt.savefig("../images/ROC_CURVE.png")
    
    
path__Rec_img = "./playground/Avengers/train/chris_hemsworth/chris_hemsworth2.png"
test_image_path = "./playground/Avengers/test"
# Call calculate_performance function with the test image path
# fpr_values, tpr_values, accuracy, precision, specificity = calculate_performance(test_image_path)


# # Print the results
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Specificity:", specificity)

# # Plot ROC curve
# ROC_plot(tpr_values, fpr_values)

unknown_face = cv2.imread(path__Rec_img)#read the image
gray = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)
unknown_face = cv2.resize(gray, (64, 64))#resize with 64*64 shape
unknown = np.array(unknown_face, dtype='float64').flatten()
train_imgs_paths, train_labels, best_match = PCA_APPLY(unknown)
best_match_img = cv2.imread(train_imgs_paths[best_match])
cv2.imshow("training", best_match_img)
cv2.imshow("test", unknown_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("best", best_match)