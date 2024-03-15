from pydantic import BaseModel


class histogramModel(BaseModel):
    type: str
    minWidth: int
    maxWidth: int
