from pydantic import BaseModel


class ThresholdingModel(BaseModel):
    type: str
    threshold: int
    blockSize: int
