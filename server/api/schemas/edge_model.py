from pydantic import BaseModel


class edgeModel(BaseModel):
    type: str
    direction: str
    kernelSize: int
    sigma: float
    lowerThreshold: int
    upperThreshold: int
