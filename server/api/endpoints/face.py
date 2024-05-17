import os

from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.config import models_folder

from api.schemas.face_model import FaceModel
from api.services.face_detection_service import FaceDetection
from api.services.face_recognition_service import FaceRecognition

router = APIRouter()

face_options = ["faceDetection", "faceRecognition"]


@router.post("/face/{image_id}")
async def detect_recognise_faces(image_id: str, face: FaceModel):
    """
    Detect or recognize faces in an image using PCA.
    """
    if face.type not in face_options:
        raise HTTPException(status_code=400, detail="This option doesn't exist.")

    image_path = get_image(image_id)
    cascade_file = os.path.join(models_folder, "haarcascade_frontalface_default.xml")
    if face.type == "faceDetection":
        face_detection = FaceDetection(cascade_file)
        output_image = face_detection.detect_and_draw_faces(image_path)
    else:
        svm_file = os.path.join(models_folder, "svm_model.pkl")
        pca_file = os.path.join(models_folder, "pca.pkl")
        face_recognition = FaceRecognition(cascade_file, svm_file, pca_file)
        output_image = face_recognition.recognize_faces(image_path)

    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Corners detection applied successfully.",
        "image": output_image,
    }
