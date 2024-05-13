import os
from fastapi import FastAPI
from fastapi_utilities import repeat_every

from api.endpoints import (
    upload,
    filter,
    noise,
    hybrid,
    thresholding,
    edge,
    histogram,
    hough,
    active_contour,
    corners,
    matching,
    sift,
    segmentation,
    face,
)
from api.config import uploads_folder


app = FastAPI(
    title="Aura API",
    description="API for Aura image processing toolkit.",
    version="0.1.0",
)


async def startup_event():
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)
    # else:
    #     for file in os.listdir(uploads_folder):
    #         os.remove(os.path.join(uploads_folder, file))


# @repeat_every(seconds=60 * 60)
# async def clear_uploads():
#     for file in os.listdir(uploads_folder):
#         os.remove(os.path.join(uploads_folder, file))


app.add_event_handler("startup", startup_event)
# app.add_event_handler("startup", clear_uploads)

app.include_router(upload.router, prefix="/api")
app.include_router(filter.router, prefix="/api")
app.include_router(noise.router, prefix="/api")
app.include_router(hybrid.router, prefix="/api")
app.include_router(thresholding.router, prefix="/api")
app.include_router(edge.router, prefix="/api")
app.include_router(histogram.router, prefix="/api")
app.include_router(hough.router, prefix="/api")
app.include_router(active_contour.router, prefix="/api")
app.include_router(corners.router, prefix="/api")
app.include_router(matching.router, prefix="/api")
app.include_router(sift.router, prefix="/api")
app.include_router(segmentation.router, prefix="/api")
app.include_router(face.router, prefix="/api")
