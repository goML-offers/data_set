from typing import List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, json, Json
# from pydantic.generics import GenericModel


class UploadFileSchema(BaseModel):
    user_id: Optional[int] = None
    file_type: str
    # folder_id: int
    file_language: str
    # file_path: str
    # org_id : Optional[int] = None
    class Config: 
        orm_mode = True

class UserIdFileIdSchema(BaseModel):
    user_id: int
    file_id: int

class UpdateExtractionSchema(BaseModel):
    extraction_id: int
    question_id : int
    user_answer: str