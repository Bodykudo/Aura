from fastapi import APIRouter, HTTPException

from api.schemas.noise_model import NoiseModel
from api.services.noise_service import Noise
from api.utils import convert_image, get_image

router = APIRouter()

noise_types = ["uniform", "gaussian", "salt_and_pepper"]


@router.post("/api/noise/{image_id}")
async def apply_noise(image_id: str, noise: NoiseModel):
    if noise.type not in noise_types:
        raise HTTPException(status_code=400, detail="Noise type doesn't exist.")

    image_path = get_image(image_id)

    noisy_image = None
    if noise.type == "uniform":
        noisy_image = Noise.uniform_noise(image_path, noise.noiseValue)
    elif noise.type == "gaussian":
        noisy_image = Noise.gaussian_noise(image_path, noise.mean, noise.variance)
    elif noise.type == "salt_and_pepper":
        noisy_image = Noise.salt_and_pepper_noise(
            image_path, noise.saltProbability, noise.pepperProbability
        )

    noisy_image = convert_image(noisy_image)

    return {
        "success": True,
        "message": "Filter applied successfully.",
        "image": noisy_image,
    }
