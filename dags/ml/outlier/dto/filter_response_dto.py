from typing import Optional

from pydantic import BaseModel, Field


class ContentResponseDTO(BaseModel):
    response: str = Field(title='Response from filtering', default='Ok')
    message: Optional[str] = Field(title='Message with error')
    outlierNum: int = Field(title='Outlier num')

    def dict(self, *args, **kwargs):
        result_dict: dict = super(ContentResponseDTO, self).dict(*args, **kwargs)
        if result_dict['message'] is None:
            result_dict.pop('message')
        return result_dict

    class Config:
        allow_mutation = False
