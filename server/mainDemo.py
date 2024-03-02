# main.py
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from filters import applyFilter
# from upload import uploadImage
# import uvicorn

# app = FastAPI(title="Upload file using FastAPI")

# @app.post("/api/upload")
# async def uploadImageRoute(file: UploadFile = File(...)):
#     try:
#         return uploadImage(file)
#     except HTTPException as e:
#         return {"success": False, "error": str(e)}
    
# @app.post("/api/filter/{imageID}")
# async def applyFilterRoute(imageID: str, filterParams: dict):
#     try:
#         return applyFilter(imageID, filterParams)
#     except HTTPException as e:
#         return {"success": False, "error": str(e)}
    
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", reload=True)