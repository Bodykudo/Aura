from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.config import uploads_folder
from api.schemas.histogram_model import histogramModel
from api.services.histogram_service import Histogram

router = APIRouter()

transform_types = ["grayscale", "normalization", "equalization"]


@router.post("/api/histogram/{image_id}")
async def apply_histogram(image_id: str, histogram: histogramModel):
    if histogram.type not in transform_types:
        raise HTTPException(status_code=400, detail="Histogram type doesn't exist")

    image_path = get_image(image_id)
    image = read_image(image_path)

    # if conditions missing

    output_histogram = None

    return {
        "success": True,
        "Message": "Histogram applied successfully.",
        "type": histogram.type,
        "histogram": output_histogram,
    }
