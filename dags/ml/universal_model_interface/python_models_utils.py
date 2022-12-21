import logging
from typing import List, Dict, Optional

import pandas as pd

from ml.universal_model_interface.data_types.types.python_model_type import PythonModelParams
from ml.universal_model_interface.model_runner.python_model_runner import PythonModelRunner
from ml.universal_model_interface.exceptions import ModelRunnerException
from ml.types import DatasetParams, NotationParams, ResearchParams
from ml.utils.sql import get_clickhouse_dataframe

loggers = {}


def get_dataframe_from_python_model(
        research_params: ResearchParams,
        dataset_params: DatasetParams,
        notation_params: NotationParams,
        python_model_params: PythonModelParams,
        columns_with_key: Optional[List[str]],
        filters: bool,
) -> pd.DataFrame:
    logger_ = logging.getLogger(__name__ + '_get_path_of_dumped_dataframe_from_python_model')

    logger_.debug(f"Reading clickhouse table: {dataset_params.path}")

    df = get_clickhouse_dataframe(
        research_params,
        dataset_params,
        columns_with_key,
        filters,
    )

    model_runner = PythonModelRunner(df, python_model_params, notation_params)
    model_runner.run()
    result = model_runner.get_result()
    if result is None:
        logger_.error("Get None from python model")
        raise ModelRunnerException("Get None from python model")

    return result
