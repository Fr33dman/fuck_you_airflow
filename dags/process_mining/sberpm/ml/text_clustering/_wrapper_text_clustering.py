from typing import Dict

from distutils.util import strtobool

from pandas import DataFrame

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.ml.text_clustering import TextClustering


class WrapperTextClustering:
    def __init__(self, data: DataFrame, model_params: Dict):
        model_params, notation_params = (
            model_params["model_params"],
            model_params["notation_params"],
        )

        if isinstance(model_params["only_unique_descriptions"], str):
            model_params["only_unique_descriptions"] = bool(strtobool(model_params["only_unique_descriptions"]))

        min_samples = min(
            2,
            int(model_params["min_samples"]),
        )

        self._model = TextClustering(
            data=DataHolder(
                data,
                id_column=notation_params["id_col"],
                activity_column=notation_params["status_col"],
                start_timestamp_column=notation_params["date_col"],
                end_timestamp_column=notation_params["date_end_col"],
            ),
            description=model_params["description"],
            pca_dim=model_params["pca_dim"],
            type_model_w2v=(model_params["type_model_w2v"] == "navec"),
            min_samples=min_samples,
            only_unique_descriptions=model_params["only_unique_descriptions"],
            cluster_marking=[],
        )

    def run_model(self) -> DataFrame:
        self._model.apply()

        return self._model.get_result()
