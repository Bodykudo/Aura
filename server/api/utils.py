import os
import cv2
import base64
from secrets import token_hex
import time
import numpy as np

from fastapi import HTTPException
from api.config import uploads_folder


def generate_image_id():
    timestamp = str(int(time.time()))[-4]
    randomPart = token_hex(2)
    imageID = f"{timestamp}{randomPart}"
    return imageID


def read_image(image_path):
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return image


def get_image(image_id):
    image_path = None

    for filename in os.listdir(uploads_folder):
        if filename.startswith(f"{image_id}."):
            image_path = os.path.join(uploads_folder, filename)
            return image_path

    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found.")


def convert_image(output_image):
    is_success, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    )

    if is_success:
        base64_image = base64.b64encode(buffer).decode("utf-8")
        return base64_image
    else:
        raise ValueError("Failed to encode the image as Base64")
