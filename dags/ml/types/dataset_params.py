from typing import Optional

from pydantic import BaseModel, validator


class DatasetParams(BaseModel):
    path: str
    date_format: Optional[str]
