from typing import Union, Optional

from pydantic import BaseModel


class PythonModelSingleParam(BaseModel):
    paramName: str
    paramAlias: str
    paramDescription: Optional[str]
    paramType: str
    nestedParamType: Optional[str] = None
    paramValue: Optional[Union[str, bool, list, int, float]] = None
    groupId: Optional[str] = None
    isMultiple: bool = False
    containsNotation: bool = False
    isNullable: bool
