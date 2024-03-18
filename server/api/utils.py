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


def read_image(image_path, grayscale=False):
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    is_success, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    )

    if is_success:
        base64_image = base64.b64encode(buffer).decode("utf-8")
        return base64_image
    else:
        raise ValueError("Failed to encode the image as Base64")


def get_image_dimensions(image):
    if len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    elif len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        raise ValueError("Unsupported image shape")

    return height, width, channels


def pad_image(image, kernel_size):
    _, _, channels = get_image_dimensions(image)

    pad = kernel_size // 2

    if channels == 1:
        padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")
    else:
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    return padded_image, pad


def convolve(image, kernel):
    height, width, channels = get_image_dimensions(image)
    padded_image, pad = pad_image(image, kernel.shape[0])

    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(channels):
        for y in range(pad, height + pad):
            for x in range(pad, width + pad):
                if channels == 1:
                    result[y - pad, x - pad] = np.sum(
                        padded_image[y - pad : y + pad + 1, x - pad : x + pad + 1]
                        * kernel
                    )
                else:
                    result[y - pad, x - pad, i] = np.sum(
                        padded_image[y - pad : y + pad + 1, x - pad : x + pad + 1, i]
                        * kernel
                    )

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_fft(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    return f_transform_shifted
