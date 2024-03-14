from fastapi import APIRouter, HTTPException

from api.utils import convert_image, get_image
from api.config import uploads_folder
from api.schemas.filter_model import FilterModel
from api.services.filter_service import Filter
from api.services.hybrid_service import Hybrid

import numpy as np

router = APIRouter()

filter_types = ["average", "gaussian", "median", "low", "high"]


@router.post("/api/filter/{image_id}")
async def applyFilter(image_id: str, filter: FilterModel):
    if filter.type not in filter_types:
        raise HTTPException(status_code=400, detail="Filter type doesn't exist.")

    image_path = get_image(image_id)

    filtered_image = None
    if filter.type == "average":
        filtered_image = Filter.average_filter(image_path, filter.kernelSize)
    elif filter.type == "gaussian":
        filtered_image = Filter.gaussian_filter(
            image_path, filter.kernelSize, filter.sigma
        )
    elif filter.type == "median":
        filtered_image = Filter.median_filter(image_path, filter.kernelSize)
    elif filter.type == "low":
        _, filtered_image = Hybrid.apply_low_pass(image_path, filter.radius)
    elif filter.type == "high":
        _, filtered_image = Hybrid.apply_high_pass(image_path, filter.radius)

    filtered_image = convert_image(filtered_image)

    return {
        "success": True,
        "message": "Filter applied successfully.",
        "image": filtered_image,
    }
