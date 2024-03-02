from fastapi import FastAPI,UploadFile,File,HTTPException
from secrets import token_hex
from pydantic import BaseModel
import uvicorn 
import os
import time

app=FastAPI(title="Upload File using FastAPI")

uploadFolder="uploads"

imageIDs=[]
filterTypes=["average", "guassian", "median"]
Filters={}

class Filter(BaseModel):
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
    imageIDs.append(imageID)
    return {"success": True, "file path":filePath,"message":"Image uploaded successfully 3ady."}
    # return imageIDs

@app.post("/api/filter/{imageID}")
async def applyFilter(imageID: str, filter: Filter):
    if imageID not in imageIDs:
        raise HTTPException(status_code=404, detail="Image ID not found.")
    if filter.type not in filterTypes:
        raise HTTPException(status_code=400, detail="Filter type doesn't exist.")
    
    Filters[imageID] = filter
    return {"success": True, "message": "Filter applied successfully 3ady."}
    # return Filters[imageID]

    
if __name__=="__main__":
    uvicorn.run("main:app",host="127.0.0.1", reload=True)

