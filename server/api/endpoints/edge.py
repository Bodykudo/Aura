from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.config import uploads_folder
from api.schemas.edge_model import EdgeModel
from api.services.edge_service import EdgeDetector

router = APIRouter()

edge_detector = ["sobel", "roberts", "prewitt", "canny"]
sobel_detector_directions = ["x", "y", "both"]


@router.post("/edge/{image_id}")
async def apply_edge(image_id: str, edge: EdgeModel):
    """
    Apply edge detection to an image.
    """
    if edge.detector not in edge_detector:
        raise HTTPException(status_code=400, detail="Edge detector doesn't exist.")

    image_path = get_image(image_id)
    output_image = None

    if edge.detector == "sobel":
        if edge.direction not in sobel_detector_directions:
            raise HTTPException(
                status_code=400, detail="Direction of edge doesn't exist."
            )
        else:
            output_image, _ = EdgeDetector.sobel_edge_detection(
                image_path,
                edge.kernelSize,
                edge.direction,
            )
    elif edge.detector == "roberts":
        output_image = EdgeDetector.roberts_edge_detection(image_path, edge.kernelSize)
    elif edge.detector == "prewitt":
        output_image = EdgeDetector.prewitt_edge_detection(image_path, edge.kernelSize)
    elif edge.detector == "canny":
        output_image = EdgeDetector.canny_edge_detection(
            image_path,
            edge.kernelSize,
            edge.sigma,
            edge.lowerThreshold,
            edge.upperThreshold,
        )

    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Edge detection applied successfully.",
        "image": output_image,
    }
