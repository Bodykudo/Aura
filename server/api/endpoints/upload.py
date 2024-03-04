import os
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.utils import generate_image_id
from api.config import uploads_folder

router = APIRouter()


@router.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    allowed_extensions = {"png", "jpg", "jpeg"}
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail="Only PNG, JPG, and JPEG files are allowed."
        )

    image_id = generate_image_id()
    filePath = os.path.join(uploads_folder, f"{image_id}.{file_extension}")
    with open(filePath, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "success": True,
        "filePath": filePath,
        "fileId": image_id,
        "message": "Image uploaded successfully.",
    }
