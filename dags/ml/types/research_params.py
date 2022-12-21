from typing import Optional

from pydantic import BaseModel


class ResearchParams(BaseModel):
    widget_id: int
    workspace_id: int
    widget_type: Optional[str] = None
