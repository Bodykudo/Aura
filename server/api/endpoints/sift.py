import time
from fastapi import APIRouter, HTTPException

from api.schemas.sift_model import SiftKeypointsModel, SiftMatchingModel
from api.services.sift_service import SIFT
from api.utils import convert_image, get_image

router = APIRouter()

matching_types = ["ssd", "ncc"]


@router.post("/sift/keypoints/{image_id}")
async def detect_keypoints(image_id: str, sift_options: SiftKeypointsModel):
    """
    Detect keypoints in an image using SIFT.
    """
    image_path = get_image(image_id)

    start_time = time.time()
    sift = SIFT(
        image_path,
        sigma=sift_options.sigma,
        num_intervals=sift_options.numIntervals,
        assumed_blur=sift_options.assumedBlur,
    )
    sift.apply()
    output = sift.draw_keypoints()
    end_time = time.time()
    elapsed_time = end_time - start_time

    output = convert_image(output)

    return {
        "success": True,
        "message": "Keypoints detected successfully.",
        "image": output,
        "time": elapsed_time,
    }


@router.post("/sift/matching")
async def match_features(matching_options: SiftMatchingModel):
    """ """
    if matching_options.type not in matching_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid matching type. Please provide a valid matching type.",
        )

    original_image_path = get_image(matching_options.originalImageId)
    template_image_path = get_image(matching_options.templateImageId)

    start_time = time.time()

    original_image_sift = SIFT(
        original_image_path,
    )
    original_image_sift.apply()
    original_image_keypoints = original_image_sift.get_keypoints()
    original_image_descriptors = original_image_sift.get_descriptors()

    template_image_sift = SIFT(
        template_image_path,
    )
    template_image_sift.apply()
    template_image_keypoints = template_image_sift.get_keypoints()
    template_image_descriptors = template_image_sift.get_descriptors()

    matched_image = SIFT.match_images(
        original_image_path,
        original_image_keypoints,
        original_image_descriptors,
        template_image_path,
        template_image_keypoints,
        template_image_descriptors,
        matching_options.type,
        matching_options.numMatches,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    output = convert_image(matched_image)

    return {
        "success": True,
        "message": "Image matching applied successfully.",
        "image": output,
        "time": elapsed_time,
    }
