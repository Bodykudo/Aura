# # filters.py
# from fastapi import HTTPException
# from upload import imageIDs,appliedFilters

# filterTypes = {"average", "gaussian", "median"}

# async def applyFilter(imageID: str, filterParams: dict):
#     if imageID not in imageIDs:
#         raise HTTPException(status_code=404, detail="Image ID not found.")
#     if filterParams.get("type") not in filterTypes:
#         raise HTTPException(status_code=400, detail="Invalid filter type.")
    
#     # Store filter parameters for the image
#     appliedFilters.setdefault(imageID, []).append(filterParams)
    
#     # Implement filter logic here

#     return {"success": True, "message": "Filter applied successfully gded."}
