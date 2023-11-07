from typing import List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, json, Json
# from pydantic.generics import GenericModel



class assetCreation(BaseModel):
    project_id: int
    user_id: str
    extraction_id_brand_image:int
    extraction_id_tone_of_voice:int
    table_name:str
    file_id_logo:int
    file_id_picture_bank_image:int
