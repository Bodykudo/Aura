from pydantic import BaseModel


class SiftKeypointsModel(BaseModel):
    sigma: float
    numIntervals: int
    assumedBlur: float


class SiftMatchingModel(BaseModel):
    type: str
    originalImageId: str
    templateImageId: str
    numMatches: int
