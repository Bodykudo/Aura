from fastapi import APIRouter, HTTPException

from api.schemas.hybrid_model import HybridModel
from api.services.hybrid_service import Hybrid
from api.utils import convert_image, get_image

router = APIRouter()

filter_types = ["high", "low"]


@router.post("/api/hybrid")
async def hybrid_image(hybrid_options: HybridModel):
    if (
        hybrid_options.firstFilterType not in filter_types
        or hybrid_options.secondFilterType not in filter_types
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filter type. Must be one of 'high', or 'low'.",
        )

    if hybrid_options.firstFilterType == hybrid_options.secondFilterType:
        raise HTTPException(
            status_code=400,
            detail="Filter types must be different to create a hybrid image.",
        )

    image_path_1 = get_image(hybrid_options.firstImageId)
    image_path_2 = get_image(hybrid_options.secondImageId)

    hybrid_image, filtered_image_1, filtered_image_2 = Hybrid.apply_mixer(
        image_path_1,
        image_path_2,
        hybrid_options.firstFilterType,
        hybrid_options.filterRadius,
    )

    filtered_image_1 = convert_image(filtered_image_1)
    filtered_image_2 = convert_image(filtered_image_2)
    hybrid_image = convert_image(hybrid_image)

    return {
        "success": True,
        "message": "Filter applied successfully.",
        "filteredImage1": filtered_image_1,
        "filteredImage2": filtered_image_2,
        "hybridImage": hybrid_image,
    }
