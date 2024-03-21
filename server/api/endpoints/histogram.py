from fastapi import APIRouter, HTTPException

from api.schemas.histogram_model import HistogramModel
from api.services.histogram_service import Histogram
from api.utils import convert_image, get_image, read_image


router = APIRouter()

transform_types = ["grayscale", "normalization", "equalization"]


@router.post("/histogram/{image_id}")
async def apply_histogram(image_id: str, histogram: HistogramModel):
    """
    Apply grayscale, normalization, or equalization to an image.
    """
    if histogram.type not in transform_types:
        raise HTTPException(status_code=400, detail="Histogram type doesn't exist")

    image_path = get_image(image_id)
    image = read_image(image_path)
    transformed_image = None

    if histogram.type == "grayscale":
        transformed_image = Histogram.grayscale_image(image)
    elif histogram.type == "normalization":
        transformed_image = Histogram.normalize_image(image)
    elif histogram.type == "equalization":
        transformed_image = Histogram.equalize_image(image)

    original_histogram = Histogram.get_histogram(image)
    original_cdf = Histogram.get_cdf(image)
    transformed_histogram = Histogram.get_histogram(
        transformed_image,
        min_range=0,
        max_range=1 if histogram.type == "normalization" else 256,
    )
    transformed_cdf = Histogram.get_cdf(
        transformed_image,
        min_range=0,
        max_range=1 if histogram.type == "normalization" else 256,
    )

    transformed_image = convert_image(
        transformed_image, is_float=histogram.type == "normalization"
    )

    return {
        "success": True,
        "image": transformed_image,
        "histogram": {
            "original": {
                "histogram": original_histogram,
                "cdf": original_cdf,
            },
            "transformed": {
                "histogram": transformed_histogram,
                "cdf": transformed_cdf,
            },
        },
    }
