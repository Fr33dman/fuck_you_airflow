from typing import Dict, Optional

from pydantic import BaseModel


class FrontPythonModelResponseParam(BaseModel):
    name: str
    value: Optional[str]


class FrontPythonModelResponse(BaseModel):
    modelLabel: str
    modelName: str
    params: Dict
