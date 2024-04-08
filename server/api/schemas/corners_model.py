from pydantic import BaseModel


class CornersModel(BaseModel):
    type: str
    blockSize: int
    kernelSize: int
    k: float
    threshold: float
    maxCorners: int
    qualityLevel: float
    minDistance: int
