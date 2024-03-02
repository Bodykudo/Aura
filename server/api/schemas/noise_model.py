from pydantic import BaseModel


class NoiseModel(BaseModel):
    type: str
    noiseValue: int
    mean: int
    variance: int
    saltProbability: float
    pepperProbability: float
