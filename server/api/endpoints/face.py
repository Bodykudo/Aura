import os
import time
import cv2

from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.config import models_folder

from api.services.face_detection_service import FaceDetection
from api.schemas.face_model import FaceModel
from api.services.face_recognition_service import FaceRecognition

router = APIRouter()

face_options = ["faceDetection", "faceRecognition"]


@router.post("/face/{image_id}")
async def detect_recognise_faces(image_id: str, face: FaceModel):
    if face.type not in face_options:
        raise HTTPException(status_code=400, detail="This option doesn't exist.")

    image_path = get_image(image_id)
    cascade_file = os.path.join(models_folder, "haarcascade_frontalface_default.xml")
    face_detection = FaceDetection(cascade_file)
    image = read_image(image_path)
    image_with_faces = face_detection.detect_and_draw_faces(image)

    output_image = None
    if face.type == "faceRecognition":
        face_recognition = FaceRecognition()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # svm_model, pca = train_svm(150)
        # save_model(
        # svm_model,
        # pca,
        # os.path.join(models_folder, "svm_model.pkl"),
        # os.path.join(models_folder, "pca.pkl"),
        # )
        svm_file = os.path.join(models_folder, "svm_model.pkl")
        pca_file = os.path.join(models_folder, "pca.pkl")
        svm_model, pca = face_recognition.load_model(svm_file, pca_file)

        for _, (x, y, w, h) in enumerate(
            face_detection.detect_faces(image_with_faces), start=0
        ):
            face_region = gray[y : y + h, x : x + w]
            ds = face_recognition.predict_face(face_region, pca, svm_model)
            cv2.rectangle(image_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = ds
            cv2.putText(
                image_with_faces,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    output_image = convert_image(image_with_faces)

    return {
        "success": True,
        "message": "Corners detection applied successfully.",
        "image": output_image,
    }
