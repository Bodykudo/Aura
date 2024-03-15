from pydantic import BaseModel


class edgeModel(BaseModel):
    detector: str
    direction: str
    kernelSize: int
    sigma: float
    lowerThreshold: int
    upperThreshold: int
