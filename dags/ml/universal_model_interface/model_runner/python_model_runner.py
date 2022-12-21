import importlib
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from ml.types import NotationParams
from ml.universal_model_interface.data_types.types.python_model_type import PythonModelParams
from ml.universal_model_interface.exceptions import ModelRunnerException


class PythonModelRunner:
    """
    Base class for running python models
    """

    def __init__(self,
                 df: pd.DataFrame,
                 model_params: PythonModelParams,
                 notation_params: NotationParams):
        """

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to use in model
        model_params : PythonModelParams
            Params for model
        notation_params : NotationParams
            Dataset notation params - id_col, date_col, etc.
        """
        self._df = df
        self._model_params = model_params
        self._notation_params = notation_params
        self._model = None
        self._result: Optional[pd.DataFrame] = None

    def _setup_model(self):
        """
        Setup model with params

        Returns
        -------

        """
        imported = importlib.import_module(self._model_params.importPath)
        try:
            python_model_params = {
                "notation_params": self._notation_params.dict(),
                "model_params":
                    {
                        x.paramName: x.paramValue
                        for x in self._model_params.modelParams
                    }
            }
            self._model = \
                getattr(imported, self._model_params.functionName)(
                    self._df,
                    python_model_params
                )

        except Exception as e:
            raise ModelRunnerException(
                f"In model: {self._model_params.modelName}, "
                f"got exception: {str(e)}"
            )

    def run(self):
        """
        Run model

        Returns
        -------
            pd.DataFrame
        """
        if self._model is None:
            self._setup_model()

        if self._result is None:
            self._result = getattr(self._model,
                                   self._model_params.methodToUse)()
            # Drop index in order to put in Spark DataFrame later
            self._result.reset_index(drop=False, inplace=True)
            # Converting Iterable to string in order to bypass Spark limitations
            # in types inferring
            row_to_check_types = self._result.iloc[0]
            cols_from_iterable_to_string = []
            cols_from_bool_to_string = []
            for col_name, val in row_to_check_types.iteritems():
                if isinstance(val, Iterable) and not isinstance(val, str):
                    cols_from_iterable_to_string.append(col_name)
                elif isinstance(val, np.bool_):
                    cols_from_bool_to_string.append(col_name)
            if len(cols_from_iterable_to_string) > 0:
                self._result.loc[:, cols_from_iterable_to_string] = \
                    self._result.loc[:, cols_from_iterable_to_string].applymap(
                        lambda x: ", ".join(x))
            if len(cols_from_bool_to_string) > 0:
                self._result.loc[:, cols_from_bool_to_string] = \
                    self._result.loc[:, cols_from_bool_to_string].astype("str")

    def get_result(self) -> Optional[pd.DataFrame]:
        """
        Get result pd.DataFrame or None from Python model.

        Returns
        -------
            pd.DataFrame or None
        """
        if self._result is None:
            self.run()
        return self._result
