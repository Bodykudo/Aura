from fastapi import APIRouter, HTTPException
from api.utils import convert_image, get_image, read_image
from api.config import uploads_folder
from api.schemas.histogram_model import histogramModel
from api.services.histogram_service import Histogram
import cv2
import numpy as np


router = APIRouter()

transform_types = ["grayscale", "normalization", "equalization"]


@router.post("/api/histogram/{image_id}")
async def apply_histogram(image_id: str, histogram: histogramModel):
    if histogram.type not in transform_types:
        raise HTTPException(status_code=400, detail="Histogram type doesn't exist")

    image_path = get_image(image_id)
    image = read_image(image_path)

    output_image = None
    if histogram.type == "grayscale":
        output_image = Histogram.convert_to_grayscale(image)
    elif histogram.type == "normalization":
        output_image = Histogram.normalize_image(image)
    elif histogram.type == "equalization":
        output_image = Histogram.equalize_histogram(image)

    new_histogram = Histogram.calcualte_histogram(
        output_image,
        min_range=0,
        max_range=1 if histogram.type == "normalization" else 256,
    )
    output_image = convert_image(
        output_image, is_float=histogram.type == "normalization"
    )
    original_histogram = Histogram.calcualte_histogram(image)
    original_red = original_histogram[2][0]
    original_green = original_histogram[1][0]
    original_blue = original_histogram[0][0]

    if np.array_equal(original_red, original_green) and np.array_equal(
        original_blue, original_green
    ):
        original_histogram_data = [
            {"name": int(i), "gray": int(original_red[i])}
            for i in range(len(original_red))
        ]
        original_cdf_data = [
            {"name": int(i), "gray": int(np.cumsum(original_red)[i])}
            for i in range(len(original_red))
        ]
    else:
        original_histogram_data = [
            {
                "name": int(i),
                "red": int(original_histogram[2][0][i]),
                "green": int(original_histogram[1][0][i]),
                "blue": int(original_histogram[0][0][i]),
            }
            for i in range(len(original_histogram[0][0]))
        ]
        original_cdf_data = [
            {
                "name": int(i),
                "red": int(np.cumsum(original_histogram[2][0])[i]),
                "green": int(np.cumsum(original_histogram[1][0])[i]),
                "blue": int(np.cumsum(original_histogram[0][0])[i]),
            }
            for i in range(len(original_histogram[0][0]))
        ]

    if histogram.type == "grayscale":
        new_histogram_data = [
            {"name": int(i), "gray": int(new_histogram[i])}
            for i in range(len(new_histogram))
        ]
        new_cdf_data = [
            {"name": int(i), "gray": int(np.cumsum(new_histogram)[i])}
            for i in range(len(new_histogram))
        ]
    else:
        new_red = new_histogram[2][0]
        new_green = new_histogram[1][0]
        new_blue = new_histogram[0][0]
        print(new_red)

        if np.array_equal(new_red, new_green) and np.array_equal(new_green, new_blue):
            new_histogram_data = [
                {"name": int(i), "gray": int(new_red[i])} for i in range(len(new_red))
            ]
            new_cdf_data = [
                {"name": int(i), "gray": int(np.cumsum(new_red)[i])}
                for i in range(len(new_red))
            ]
        else:
            new_histogram_data = [
                {
                    "name": int(i),
                    "red": int(new_histogram[2][0][i]),
                    "green": int(new_histogram[1][0][i]),
                    "blue": int(new_histogram[0][0][i]),
                }
                for i in range(len(new_histogram[0][0]))
            ]
            new_cdf_data = [
                {
                    "name": int(i),
                    "red": int(np.cumsum(new_histogram[2][0])[i]),
                    "green": int(np.cumsum(new_histogram[1][0])[i]),
                    "blue": int(np.cumsum(new_histogram[0][0])[i]),
                }
                for i in range(len(new_histogram[0][0]))
            ]

    return {
        "success": True,
        "image": output_image,
        "histogram": {
            "original": {
                "histogram": original_histogram_data,
                "cdf": original_cdf_data,
            },
            "new": {
                "histogram": new_histogram_data,
                "cdf": new_cdf_data,
            },
        },
    }
