from typing import Optional

from pydantic import BaseModel


class NotationParams(BaseModel):
    id_col: str
    date_col: Optional[str]
    date_end_col: Optional[str]
    status_col: str
    user_id_col: Optional[str]
