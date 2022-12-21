from typing import List, Optional
from pandas import DataFrame

from ml.types import ResearchParams, DatasetParams
from ml.db import ClickhouseClient


def get_sql_query_from_params(
        filter_c_is_empty: bool,
        research_params: ResearchParams,
        dataset_params: DatasetParams,
        columns: Optional[List[str]],
        sample: int = 0
) -> str:
    """
    Return valid SQL expression formed from research_params & dataset_params
    Parameters
    ----------
    filter_c_is_empty : bool
        Is filterC empty table or not
    research_params : ResearchParams
    dataset_params : DatasetParams
    columns: list[str]

    Returns
    -------
        sql_query : str
            valid SQL expression formed from research_params & dataset_params
    """
    if sample > 0:
        sample_query = f'SAMPLE {sample}'
    need_columns = ', '.join(columns) if columns else '*'
    if filter_c_is_empty:
        sql_query = \
            f"""
                SELECT {need_columns} FROM {dataset_params.path}a {sample_query}
            """
    else:
        if "TEXT_ANALYSIS" == research_params.widget_type:
            if 0 != research_params.widget_id:
                sql_query = \
                    f"""
                        SELECT {need_columns} FROM {dataset_params.path}a {sample_query}
                        WHERE _caseId IN (
                            SELECT _caseId FROM {dataset_params.path}filterC 
                            WHERE _workspaceId = {research_params.workspace_id}
                                AND _widgetId = {research_params.widget_id}
                                AND _widgetType = 'TEXT_ANALYSIS'
                        )
                    """
            else:
                sql_query = \
                    f"""
                        SELECT {need_columns} FROM {dataset_params.path}a {sample_query}
                        WHERE _caseId IN (
                            SELECT _caseId FROM {dataset_params.path}filterC 
                            WHERE _workspaceId = {research_params.workspace_id}
                                AND _widgetId = 0
                                AND _widgetType = ''
                        )
                    """
        else:
            sql_query = \
                f"""
                    SELECT {need_columns} FROM {dataset_params.path}a {sample_query}
                    WHERE _caseId IN (
                        SELECT _caseId FROM {dataset_params.path}filterC 
                        WHERE _workspaceId = {research_params.workspace_id}
                    )
                """

    return sql_query


def get_clickhouse_dataframe(
        research_params: ResearchParams,
        dataset_params: DatasetParams,
        columns_with_key: Optional[List[str]],
        filters: bool
) -> DataFrame:

    client = ClickhouseClient()

    filter_c_is_empty = ~filters

    sample_columns = 1_000_000

    query = get_sql_query_from_params(
        filter_c_is_empty,
        research_params,
        dataset_params,
        columns_with_key,
        sample_columns,
    )

    sdf = client.read_table(
        query,
        columns_with_key=columns_with_key,
    )

    return sdf
