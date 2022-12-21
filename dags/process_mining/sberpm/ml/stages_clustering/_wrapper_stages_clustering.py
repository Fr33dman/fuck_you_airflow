from typing import Mapping

from pandas import DataFrame

from process_mining.sberpm.ml.stages_clustering import StagesClustering
from process_mining.sberpm._holder import DataHolder


class WrapperStagesClustering:
    def __init__(self, data: DataFrame, model_params: Mapping):
        model_params, notation_params = (
            model_params["model_params"],
            model_params["notation_params"],
        )

        generalizing_ability = float(model_params["generalizing_ability"])
        if generalizing_ability > 1 or generalizing_ability < 0:
            generalizing_ability = 0.5

        self._model = StagesClustering(
            DataHolder(
                data,
                id_column=notation_params["id_col"],
                activity_column=notation_params["status_col"],
                start_timestamp_column=notation_params["date_col"],
                end_timestamp_column=notation_params["date_end_col"],
            ),
            stages_col=model_params["stages_col"],
            generalizing_ability=generalizing_ability,
            type_model_w2v=(model_params["type_model_w2v"] == "navec"),
        )

    def run_model(self) -> DataFrame:
        self._model.apply()

        return self._model.get_clustered_result()
