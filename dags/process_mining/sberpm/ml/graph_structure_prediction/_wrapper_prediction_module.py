from distutils.util import strtobool
from typing import Dict

from pandas import DataFrame
from pandas.api.types import is_float_dtype

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.ml.graph_structure_prediction import GSPredictor

FLOAT_PRECISION = 6


def _ensure_datetime_formatted_without_timezone(data_with_dt_index: DataFrame):
    datetime_index = data_with_dt_index.index

    if datetime_index.tz:
        data_with_dt_index.index = datetime_index.tz_convert(None)

    return data_with_dt_index


def _guarantee_boolean_model_params(model_params: dict, boolean_params=("two_models", "refit", "quick_mod")):
    for boolean_param in boolean_params:
        if isinstance(model_params[boolean_param], str):
            model_params[boolean_param] = bool(strtobool(model_params[boolean_param]))

    return model_params


def _save_nodes_probabilities(gsp_model: GSPredictor):
    result = gsp_model.nodes_prob_pred.copy()

    # * excel can't write timezones; will also drop trailing zero values in dt format
    result = _ensure_datetime_formatted_without_timezone(result)

    for col in result.columns:
        result.rename(columns={col: f"{col}_prob"}, inplace=True)

    return result


def _save_nodes_duration_prediction(gsp_model: GSPredictor):
    res_duration = gsp_model.nodes_duration_pred.copy()

    # * excel can't write timezones; will also drop trailing zero values in dt format
    res_duration = _ensure_datetime_formatted_without_timezone(res_duration)

    for col in res_duration.columns:
        res_duration.rename(columns={col: f"{col}_duration (секунд)"}, inplace=True)

    return res_duration


class WrapperGSPredictor:
    def __init__(self, data: DataFrame, model_params: Dict):
        print(f"Model params: {model_params['model_params']}")

        model_params, notation_params = (
            model_params["model_params"],
            model_params["notation_params"],
        )

        model_params = _guarantee_boolean_model_params(model_params)

        self._model = GSPredictor(
            DataHolder(
                data,
                id_column=notation_params["id_col"],
                activity_column=notation_params["status_col"],
                start_timestamp_column=notation_params["date_col"],
                end_timestamp_column=notation_params["date_end_col"],
            ),
            test_size=model_params["test_size"],
            val_size=model_params["val_size"],
            pred_period=model_params["pred_period"],
            period_type=model_params["period_type"],
            two_models=model_params["two_models"],
            edges=False,  # * keep nodes only, omit edges as slowest
            refit=model_params["refit"],
            quick_mod=model_params["quick_mod"],
        )

    def run_model(self) -> DataFrame:
        self._model.apply()

        # * keep nodes only, omit edges as slowest
        result = _save_nodes_probabilities(self._model).join(
            other=_save_nodes_duration_prediction(self._model), how="right"
        )

        result = result.round(decimals=FLOAT_PRECISION)

        for col, dtype in result.dtypes.iteritems():
            if is_float_dtype(dtype):
                result[col] = result[col].astype(str).str.replace(".", ",", regex=False)

        return result
