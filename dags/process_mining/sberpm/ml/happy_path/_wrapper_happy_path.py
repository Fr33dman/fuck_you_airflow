from typing import Dict, Any

from pandas import DataFrame

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.ml.happy_path import HappyPath


class WrapperHappyPath:
    def __init__(self, df: DataFrame, model_params: Dict[str, Any]):
        model_params, notation_params = model_params["model_params"], model_params["notation_params"]

        self._model = HappyPath(
            data_holder=DataHolder(
                df,
                id_column=notation_params["id_col"],
                activity_column=notation_params["status_col"],
                start_timestamp_column=notation_params["date_col"],
                end_timestamp_column=notation_params["date_end_col"],
            ),
            key_node=model_params["key_node"],
            mode=model_params["mode"],
            initial_state=model_params["initial_state"],
            end=model_params["end"],
            reward_for_key=model_params["reward_for_key"],
            reward_for_end=model_params["reward_for_end"],
            prob_increase=model_params["prob_increase"],
            clear_outliers=model_params["clear_outliers"],
            short_path=model_params["short_path"],
            gamma=model_params["gamma"],
            regime=model_params["regime"],
            penalty=model_params["penalty"],
            time_break=model_params["time_break"],
            output_algo_params=model_params["output_algo_params"],
        )

    def run_model(self) -> DataFrame:
        """
        Returns:
            result: DataFrame - happy path result
        """
        self._model.apply()

        result = self._model.get_df_rl()

        result["rew"] = result["rew"].astype(str).replace(".", ",")
        result.rename(columns={"rew": "reward"}, inplace=True)

        return result.to_frame()
