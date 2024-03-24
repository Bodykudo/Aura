from fastapi import APIRouter, HTTPException

from api.schemas.hough_model import HoughModel
from api.services.hough_service import Hough
from api.utils import convert_image, get_image, hex_to_rgb

router = APIRouter()

hough_types = ["lines", "circles"]


@router.post("/hough/{image_id}")
async def apply_hough_transform(image_id: str, hough: HoughModel):
    """
    Apply local/global thresholding to an image.
    """
    if hough.type not in hough_types:
        raise HTTPException(status_code=400, detail="Hough type doesn't exist.")

    image_path = get_image(image_id)

    output_image = None
    color = hex_to_rgb(hough.color)
    if hough.type == "lines":
        output_image = Hough.detect_lines(
            image_path, hough.rho, hough.theta, hough.threshold, color
        )
    elif hough.type == "circles":
        output_image = Hough.detect_circles(
            image_path, hough.minRadius, hough.maxRadius, hough.threshold, color
        )

    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Hough transform applied successfully.",
        "image": output_image,
    }
