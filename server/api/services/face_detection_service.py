import cv2

from api.utils import read_image


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

    def detect_and_draw_faces(self, image_path: str):
        image = read_image(image_path)
        faces = self.detect_faces(image)
        image_with_faces = self.draw_faces(image.copy(), faces)
        return image_with_faces
