from pydantic import BaseModel


class ActiveContourModel(BaseModel):
    centerX: float
    centerY: float
    radius: float
    iterations: int
    points: int
    windowSize: int
    alpha: float
    beta: float
    gamma: float
