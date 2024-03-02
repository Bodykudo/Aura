# # upload.py
# import os
# import time
# from secrets import token_hex
# from fastapi import UploadFile, HTTPException

# uploadFolder = "uploads"

# if not os.path.exists(uploadFolder):
#     os.makedirs(uploadFolder)

# imageIDs = set()
# appliedFilters = {}

# async def uploadImage(file: UploadFile):
#     allowedExt = {"png", "jpg", "jpeg"}
#     imageExt = file.filename.split(".")[-1].lower()
#     if imageExt not in allowedExt:
#         raise HTTPException(status_code=400, detail="Only PNG, JPG, and JPEG files are allowed.")
    
#     imageID = generateImageID()
#     imagePath = os.path.join(uploadFolder, f"{imageID}.{imageExt}")

#     with open(imagePath, "wb") as f:
#         content = await file.read()
#         f.write(content)

#     imageIDs.add(imageID)
#     return {"success": True, "image path": imagePath, "message": "Image uploaded successfully gded."}

# def generateImageID():
#     timeStamp = str(int(time.time()))[-4]
#     randomPart = token_hex(2)
#     imageID = f"{timeStamp}{randomPart}"
#     return imageID
