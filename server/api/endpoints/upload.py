import os
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.utils import generate_image_id
from api.config import uploads_folder

router = APIRouter()


@router.post("/api/upload")
async def uploadImage(file: UploadFile = File(...)):
    allowedExt = {"png", "jpg", "jpeg"}
    fileExt = file.filename.split(".")[-1].lower()
    if fileExt not in allowedExt:
        raise HTTPException(
            status_code=400, detail="Only PNG, JPG, and JPEG files are allowed."
        )

    imageID = generate_image_id()
    filePath = os.path.join(uploads_folder, f"{imageID}.{fileExt}")
    with open(filePath, "wb") as f:
        content = await file.read()
        f.write(content)
    return {
        "success": True,
        "filePath": filePath,
        "fileId": imageID,
        "message": "Image uploaded successfully.",
    }
