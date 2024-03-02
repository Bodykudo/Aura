from fastapi import APIRouter, HTTPException

from api.utils import convert_image, get_image
from api.config import uploads_folder
from api.schemas.filter_model import FilterModel
from api.services.filter_service import Filter

router = APIRouter()

filter_types = ["average", "guassian", "median"]


@router.post("/api/filter/{image_id}")
async def applyFilter(image_id: str, filter: FilterModel):
    if filter.type not in filter_types:
        raise HTTPException(status_code=400, detail="Filter type doesn't exist.")

    image_path = get_image(image_id)

    filtered_image = None
    if filter.type == "average":
        filtered_image = Filter.apply_avg_filter(image_path, filter.kernelSize)
    elif filter.type == "guassian":
        filtered_image = Filter.apply_gaussian_filter(
            image_path, filter.kernelSize, filter.sigma
        )
    elif filter.type == "median":
        filtered_image = Filter.apply_median_filter(image_path, filter.kernelSize)

    filtered_image = convert_image(filtered_image)

    return {
        "success": True,
        "message": "Filter applied successfully.",
        "image": filtered_image,
    }
