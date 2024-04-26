from fastapi import APIRouter, HTTPException

from api.schemas.thresholding_model import ThresholdingModel
from api.services.thresholding_service import Thresholding
from api.utils import convert_image, get_image

router = APIRouter()

thresholding_types = ["optimal", "otsu", "spectral"]
thresholding_scopes = ["local", "global"]


@router.post("/thresholding/{image_id}")
async def apply_thresholding(image_id: str, thresholding: ThresholdingModel):
    """
    Apply local/global thresholding to an image.
    """

    if thresholding.type not in thresholding_types:
        raise HTTPException(status_code=400, detail="Thresholding type doesn't exist.")
    if thresholding.scope not in thresholding_scopes:
        raise HTTPException(status_code=400, detail="Thresholding scope doesn't exist.")

    image_path = get_image(image_id)
    thresholded_image = None

    if thresholding.type == "optimal":
        pass
    elif thresholding.type == "otsu":
        pass
    elif thresholding.type == "spectral":

        if thresholding.scope == "global":
            thresholded_image = Thresholding.spectral_thresholding(image_path)
        elif thresholding.scope == "local":
            thresholded_image = Thresholding.spectral_thresholding_local(
                image_path, window_size=thresholding.windowSize
            )

    thresholded_image = convert_image(thresholded_image)

    return {
        "success": True,
        "message": "Thresholding applied successfully.",
        "type": thresholding.type,
        "scope": thresholding.scope,
        "image": thresholded_image,
        "image id": image_id,
    }
