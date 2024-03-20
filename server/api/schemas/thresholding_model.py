from pydantic import BaseModel


class ThresholdingModel(BaseModel):
    type: str
    threshold: int
    thresholdMargin: int
    blockSize: int
