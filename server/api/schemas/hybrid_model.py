from pydantic import BaseModel


class HybridModel(BaseModel):
    firstImageId: str
    secondImageId: str
    firstFilterType: str
    secondFilterType: str
    filterRadius: int
