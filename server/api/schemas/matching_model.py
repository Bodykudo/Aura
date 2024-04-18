from pydantic import BaseModel


class MatchingModel(BaseModel):
    type: str
    originalImageId: str
    templateImageId: str
