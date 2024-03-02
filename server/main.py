from fastapi import FastAPI,UploadFile,File,HTTPException
from secrets import token_hex
from pydantic import BaseModel
import uvicorn 
import os
import time
from Filter import Filter
from utils import functions as f

app=FastAPI(title="Upload File using FastAPI")

uploadFolder="uploads"

imageIDs=[]
filterTypes=["average", "guassian", "median"]
Filters={}

class FilterModel(BaseModel):
    type:str
    kernelSize:int
    sigma:float

if not os.path.exists(uploadFolder):
    os.makedirs(uploadFolder)

def generateImageID():
    timestamp=str(int(time.time()))[-4]
    randomPart=token_hex(2)
    imageID=f"{timestamp}{randomPart}"
    return imageID

@app.post("/api/upload")
async def uploadImage(file:UploadFile=File(...)):
    allowedExt={'png','jpg','jpeg'}
    fileExt=file.filename.split(".")[-1].lower()
    if fileExt not in allowedExt:
        raise HTTPException(status_code=400,detail="Only PNG, JPG, and JPEG files are allowed.")
    
    imageID=generateImageID()
    filePath=os.path.join(uploadFolder,f"{imageID}.{fileExt}")
    with open(filePath,"wb") as f:
        content=await file.read()
        f.write(content)
    return {"success": True, "filePath":filePath, "fileId": imageID, "message":"Image uploaded successfully."}

@app.post("/api/filter/{imageID}")
async def applyFilter(imageID: str, filter: FilterModel):
    uploadFolder = os.path.abspath('uploads')
    image_path = None
    
    for filename in os.listdir(uploadFolder):
        if filename.startswith(f"{imageID}."):
            image_path = os.path.join(uploadFolder, filename)
            break

    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    
    if filter.type not in filterTypes:
        raise HTTPException(status_code=400, detail="Filter type doesn't exist.")
    
    result = None
    if filter.type == "average":
        test = Filter.apply_avg_filter(image_path, filter.kernelSize)
        result = f.convert_image(test)

    return {"success": True, "message": "Filter applied successfully.", "output": result}
    
if __name__=="__main__":
    uvicorn.run("main:app",host="127.0.0.1", reload=True)

