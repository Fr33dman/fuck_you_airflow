from distutils.util import strtobool

from pandas import DataFrame
from pandas.api.types import is_float_dtype

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.ml.factor_analysis import FactorAnalysis

FLOAT_PRECISION = 6


class WrapperFactorAnalysis:
    def __init__(self, data: DataFrame, model_params: dict):
        if isinstance(model_params["model_params"]["extended_search"], str):
            model_params["model_params"]["extended_search"] = bool(
                strtobool(model_params["model_params"]["extended_search"])
            )
        if isinstance(model_params["model_params"]["count_others"], str):
            model_params["model_params"]["count_others"] = bool(
                strtobool(model_params["model_params"]["count_others"])
            )

        self._model = FactorAnalysis(
            DataHolder(
                data,
                id_column=model_params["notation_params"]["id_col"],
                activity_column=model_params["notation_params"]["status_col"],
                start_timestamp_column=model_params["notation_params"]["date_col"],
                end_timestamp_column=model_params["notation_params"]["date_end_col"],
                utc=True,  # * FA somehow needs utc just cause of temp design flaw
            ),
            target_column=model_params["model_params"]["target_column"],
            type_of_target=model_params["model_params"]["type_of_target"],
            categorical_cols=model_params["model_params"]["categorical_cols"],
            numeric_cols=model_params["model_params"]["numeric_cols"],
            date_cols=model_params["model_params"]["date_cols"],
            extended_search=model_params["model_params"]["extended_search"],
            count_others=model_params["model_params"]["count_others"],
        )

    def run_model(self) -> DataFrame:
        result = self._model.apply()

        result = result.round(decimals=FLOAT_PRECISION)

        for col, dtype in result.dtypes.iteritems():
            if is_float_dtype(dtype):
                result[col] = result[col].astype(str).str.replace(".", ",", regex=False)

        return result
