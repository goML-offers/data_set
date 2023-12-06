from typing import List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, json, Json
# from pydantic.generics import GenericModel



class assetCreation(BaseModel):
    project_id: int
    user_id: str
    extraction_id_brand_document:int
    extraction_id_tone_of_voice:Optional[int]=None
    table_name:str
    file_id_logo: Optional[int] = None
    file_id_picture_bank_image: Optional[int] = None



class totalAsset(BaseModel):
    table_name:str
