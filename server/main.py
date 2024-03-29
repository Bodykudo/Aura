import os
from fastapi import FastAPI

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


app.add_event_handler("startup", startup_event)

app.include_router(upload.router, prefix="/api")
app.include_router(filter.router, prefix="/api")
app.include_router(noise.router, prefix="/api")
app.include_router(hybrid.router, prefix="/api")
app.include_router(thresholding.router, prefix="/api")
app.include_router(edge.router, prefix="/api")
app.include_router(histogram.router, prefix="/api")
app.include_router(hough.router, prefix="/api")
app.include_router(active_contour.router, prefix="/api")
