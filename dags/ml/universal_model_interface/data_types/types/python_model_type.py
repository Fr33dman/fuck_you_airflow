from typing import List, Optional

from pydantic import BaseModel

from .python_model_param_type import PythonModelSingleParam


class PythonModelParams(BaseModel):
    modelName: str
    modelLabel: str
    modelAlias: str
    modelDescription: Optional[str]
    outputDescription: str
    importPath: str
    functionName: str
    methodToUse: str
    modelParams: List[PythonModelSingleParam]
