from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.config import uploads_folder
from api.schemas.edge_model import edgeModel
from api.services.edge_service import EdgeDetector

router = APIRouter()

edge_types = ["sobel", "roberts", "prewitt", "canny"]
sobel_directions = ["horizontal", "vertical", "combined"]


@router.post("/api/edge/{image_id}")
async def apply_edge(image_id: str, edge: edgeModel):
    if edge.type not in edge_types:
        raise HTTPException(status_code=400, detail="Edge type doesn't exist.")

    image_path = get_image(image_id)
    image = read_image(image_path)

    output_image = None

    if edge.type == "sobel":
        if edge.direction not in sobel_directions:
            raise HTTPException(
                status_code=400, detail="Direction of edge doesn't exist."
            )
        else:
            output_image = EdgeDetector.sobel_edge_detection(
                image,
                edge.kernelSize,
                edge.direction,
            )
    elif edge.type == "roberts":
        output_image = EdgeDetector.roberts_edge_detection(image, edge.kernelSize)
    elif edge.type == "prewitt":
        output_image = EdgeDetector.prewitt_edge_detection(image, edge.kernelSize)
    elif edge.type == "canny":
        output_image = EdgeDetector.canny_edge_detection(
            image,
            edge.kernelSize,
            edge.sigma,
            edge.lowerThreshold,
            edge.upperThreshold,
        )

    output_image = convert_image(output_image)

    return {
        "success": True,
        "message": "Edge applied successfully.",
        "type": edge.type,
        "image": output_image,
    }
