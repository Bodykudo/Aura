from fastapi import APIRouter

from api.schemas.active_contour_model import ActiveContourModel
from api.services.active_contour_service import ActiveContourService
from api.utils import convert_image, get_image

router = APIRouter()


@router.post("/active-contour/{image_id}")
async def apply_active_contour(image_id: str, contour: ActiveContourModel):
    """
    Apply local/global thresholding to an image.
    """
    image_path = get_image(image_id)

    output_image = None
    center = (int(contour.centerX), int(contour.centerY))
    output_image, perimeter, area = ActiveContourService.active_contour(
        image_path,
        center,
        int(contour.radius),
        contour.iterations,
        contour.points,
        contour.windowSize,
        contour.alpha,
        contour.beta,
        contour.gamma,
    )
    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Active contour applied successfully.",
        "image": output_image,
        "perimeter": perimeter,
        "area": area,
    }
