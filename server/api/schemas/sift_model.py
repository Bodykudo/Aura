from pydantic import BaseModel


class SIFTModel(BaseModel):
    firstImageId: str
    secondImageId: str
    octaveNumber: int
    scalesNumber: int
    sigma: float
    downsamplingFactor: float
