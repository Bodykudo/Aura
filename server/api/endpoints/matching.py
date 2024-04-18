import time
from fastapi import APIRouter, HTTPException

from api.schemas.matching_model import MatchingModel
from api.services.matching_service import Matching
from api.utils import convert_image, get_image

router = APIRouter()

matching_types = ["ssd", "ncc"]


@router.post("/matching")
async def match_images(matching_options: MatchingModel):
    """
    Match two images using the specified matching type (SSD or NCC).
    """
    if matching_options.type not in matching_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid matching type. Please provide a valid matching type.",
        )

    original_image_path = get_image(matching_options.originalImageId)
    template_image_path = get_image(matching_options.templateImageId)

    matched_image = None
    start_time = time.time()
    if matching_options.type == "ssd":
        matched_image = Matching.ssd_match(original_image_path, template_image_path)
    elif matching_options.type == "ncc":
        matched_image = Matching.cross_correlation_match(
            original_image_path, template_image_path
        )
    end_time = time.time()
    elapsed_time = end_time - start_time

    matched_image = convert_image(matched_image)

    return {
        "success": True,
        "message": "Image matching applied successfully.",
        "image": matched_image,
        "time": elapsed_time,
    }
