from pandas import DataFrame
from pandas.api.types import is_float_dtype

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.autoinsights import AutoInsights

FLOAT_PRECISION = 6


def fin_effect_summary_to_dataframe(summary):
    summary_list = summary.strip().splitlines()
    separator_indices = [-1] + [sep for sep, string in enumerate(summary_list) if string == ""] + [None]

    fin_effect_cluster_pairs = [
        summary_list[pair_start + 1 : pair_end]
        for pair_start, pair_end in zip(separator_indices, separator_indices[1:])
    ]

    stripped_cluster_pairs = [[pair[0][:-1], pair[1][1:]] for pair in fin_effect_cluster_pairs]

    return DataFrame(stripped_cluster_pairs, columns=["Финансовый эффект метрики", "Проблемные этапы"])


class WrapperAutoInsights:
    def __init__(self, data: DataFrame, model_params: dict):
        self._model = AutoInsights(
            data_holder=DataHolder(
                data,
                id_column=model_params["notation_params"]["id_col"],
                activity_column=model_params["notation_params"]["status_col"],
                start_timestamp_column=model_params["notation_params"]["date_col"],
                end_timestamp_column=model_params["notation_params"]["date_end_col"],
                text_column=model_params["model_params"]["text_column"],
            ),
            success_activity=model_params["model_params"]["success_activity"],
            cluster_eps=model_params["model_params"]["cluster_eps"],
            min_cost=model_params["model_params"]["min_cost"],
        )
        self.output_type = model_params["model_params"]["output_type"]

    def run_model(self) -> DataFrame:
        """
        output_type: Literal[
            "clustered",
            "clustered_binary",
            "financial_effect_summary",
            "transitions",
        ] = "clustered_result",

        Returns:
            result: Union[DataFrame, string] - table text description of insights
        """
        self._model.apply()

        if self.output_type == "clustered":
            result = self._model.get_clustered_result()
        elif self.output_type == "clustered_binary":
            result = self._model.get_boolean_clustered_result()
        elif self.output_type == "financial_only_effect":
            result = fin_effect_summary_to_dataframe(self._model.get_description())
        elif self.output_type == "transitions":
            result = self._model.get_transition_ai()

        result = result.round(decimals=FLOAT_PRECISION)

        for col, dtype in result.dtypes.iteritems():
            if is_float_dtype(dtype):
                result[col] = result[col].astype(str).str.replace(".", ",", regex=False)

        return result
