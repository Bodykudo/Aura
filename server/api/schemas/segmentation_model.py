from pydantic import BaseModel
from typing import List, Dict


class Point(BaseModel):
    x: float
    y: float


class SegmentationModel(BaseModel):
    type: str
    k: int
    maxIterations: int
    windowSize: int
    threshold: int
    clustersNumber: int
    colorThreshold: int
    seedPoints: List[Point]
