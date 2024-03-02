from pydantic import BaseModel


class FilterModel(BaseModel):
    type: str
    kernelSize: int
    sigma: float
