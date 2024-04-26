from pydantic import BaseModel


class ThresholdingModel(BaseModel):
    type: str
    scope: str
    windowSize: int
