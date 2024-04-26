from pydantic import BaseModel


class SegmentationModel(BaseModel):
    type: str
    k: int
    maxIterations: int
    windowSize: int
    threshold: int
    clustersNumber: int
    colorThreshold: int
    neighboursNumber: int
