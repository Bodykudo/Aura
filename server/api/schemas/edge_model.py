from pydantic import BaseModel


class EdgeModel(BaseModel):
    detector: str
    direction: str
    kernelSize: int
    sigma: float
    lowerThreshold: int
    upperThreshold: int
