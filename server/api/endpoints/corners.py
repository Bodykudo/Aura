import time
from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.schemas.corners_model import CornersModel
from api.services.corners_service import CornersDetector

router = APIRouter()

corner_detectors = ["harris", "lambda", "both"]


@router.post("/corners/{image_id}")
async def detect_corners(image_id: str, corners: CornersModel):
    """
    Detect corners in an image using Harris corner detection, Shi-Tomasi (Lambda) corner detection, or both.
    """
    if corners.type not in corner_detectors:
        raise HTTPException(status_code=400, detail="Corners detector doesn't exist.")

    image_path = get_image(image_id)

    start_time = time.time()

    output_image = CornersDetector.detect_corners(
        image_path=image_path,
        harris_params=(
            {
                "block_size": corners.blockSize,
                "kernel_size": corners.kernelSize,
                "k": corners.k,
                "threshold": corners.threshold,
            }
            if corners.type in ["harris", "both"]
            else {}
        ),
        lambda_params=(
            {
                "kernel_size": corners.kernelSize,
                "max_corners": corners.maxCorners,
                "quality_level": corners.qualityLevel,
                "min_distance": corners.minDistance,
            }
            if corners.type in ["lambda", "both"]
            else {}
        ),
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Corners detection applied successfully.",
        "time": elapsed_time,
        "image": output_image,
    }
