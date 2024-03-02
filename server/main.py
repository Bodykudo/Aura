from fastapi import FastAPI
from api.endpoints import upload, filter, noise
from api.config import uploads_folder
import os

app = FastAPI(
    title="Aura API",
    description="API for Aura image processing toolkit.",
    version="0.1.0",
)


async def startup_event():
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)


app.add_event_handler("startup", startup_event)

app.include_router(upload.router)
app.include_router(filter.router)
app.include_router(noise.router)
