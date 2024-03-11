from fastapi import APIRouter, HTTPException

from api.utils import convert_image, get_image
from api.schemas.thresholding_model import ThresholdingModel
from api.services.thresholding_service import Thresholding

router = APIRouter()

thresholding_types = ["local", "global"]


@router.post("/api/thresholding/{image_id}")
async def apply_thresholding(image_id: str, thresholding: ThresholdingModel):
    if thresholding.type not in thresholding_types:
        raise HTTPException(status_code=400, detail="Thresholding type doesn't exist.")

    image_path = get_image(image_id)

    output_image = None
    if thresholding.type == "local":
        output_image = Thresholding.local_thresholding(
            image_path, thresholding.blockSize
        )
    elif thresholding.type == "global":
        output_image = Thresholding.global_thresholding(
            image_path, thresholding.threshold
        )

    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Thresholding applied successfully.",
        "image": output_image,
    }