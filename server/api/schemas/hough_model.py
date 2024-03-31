from pydantic import BaseModel


class HoughModel(BaseModel):
    type: str
    theta: int
    threshold: int
    minRadius: int
    maxRadius: int
    color: str
    minMajorAxis: int
