from typing import Dict
import pandas as pd

from sqlalchemy.engine import Connection
from requests.sessions import Session
from sqlalchemy import create_engine


class ClickhouseClient:
    """High-level clickhouse client to work with native python clickhouse-driver
    (https://clickhouse-driver.readthedocs.io/en/latest/index.html)"""

    def __init__(self):
        self.connection: Connection = create_engine(
            f'clickhouse+http://default:!QAZ1qaz'
            f'@10.53.222.170:8123/default',
            connect_args={'http_session': Session()}
        ).connect()

    def close(self):
        """
        Close connection to ClickHouse
        Returns
        -------

        """
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def dump_pandas_dataframe(self, data: pd.DataFrame, path_to_dump: str,
                              **kwargs):
        raise NotImplementedError

    def dump_string(self, string_for_dump, filepath_for_dump):
        raise NotImplementedError

    def read_table(self,
                   path_to_read_or_query: str,
                   **kwargs) -> pd.DataFrame:
        """
        Read table from ClickHouse to pandas Dataframe

        Parameters
        ----------
        path_to_read_or_query : str
            SQL query for ClickHouse
        kwargs : Dict
            Additional params
        Returns
        -------
            pd.Dataframe
        """

        specified_columns = kwargs.get("columns_with_key")
        result = pd.read_sql(
            path_to_read_or_query,
            con=self.connection,
            columns=specified_columns
        )

        return result
