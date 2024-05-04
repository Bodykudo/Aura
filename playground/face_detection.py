import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path: str, grayscale=False):
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        original_image = cv2.imread(image_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return image


class FaceDetection:
    def __init__(self, face_cascade_path: str):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)
        return faces

    def draw_faces(self, image, faces):
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image

    def detect_and_draw_faces(self, image):
        faces = self.detect_faces(image)
        image_with_faces = self.draw_faces(image.copy(), faces)
        return image_with_faces


face_cascade_path = "haarcascade_frontalface_default.xml"
image_path = "people.jpeg"
image = read_image(image_path)
face_detection = FaceDetection(face_cascade_path)
image_with_faces = face_detection.detect_and_draw_faces(image)
plt.imshow(image_with_faces)
plt.axis("off")
plt.show()
