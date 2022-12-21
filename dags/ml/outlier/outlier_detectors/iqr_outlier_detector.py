import logging
from typing import Optional

from sqlalchemy.engine import Connection

from ml.outlier.outlier_detectors.base_outlier_detector import BaseOutlierDetector
from ml.outlier.outlier_detectors.outlier_detector_params import IQROutlierDetectorParams, OutlierDetectorTypeEnum
from ml.outlier.outlier_detectors.sql_templates import OutlierSQLTemplate, \
    PrepareTableSQLTemplate


class IQROutlierDetector(BaseOutlierDetector):
    """Inter quartile range outlier detector"""

    def __init__(self, params: IQROutlierDetectorParams):
        self.params = params

    @staticmethod
    def _execute_query(query: str, conn: Connection):
        """
        Execute query with python clickhouse-driver
        Parameters
        ----------
        query : str
            Query for ClickHouse
        conn : Connection
            DBAPI connection to ClickHouse
        Returns
        -------

        """
        conn.execute(query)

    def apply(
        self,
        database: str,
        table_name: str,
        outlier_num: int,
        exclude_process: bool,
        use_case_branch: bool,
        first_groupby: str,
        second_groupby: Optional[str],
        metric_type: str,
        min_row_count: int,
        conn: Connection,
    ):
        logger = logging.getLogger('outliers')

        alter_table_query = PrepareTableSQLTemplate(
            database,
            table_name,
            outlier_num,
        ).get_alter_table_template()
        self._execute_query(alter_table_query, conn)

        sql_template = OutlierSQLTemplate(
            outlier_detector_params=self.params,
            database=database,
            table_name=table_name,
            outlier_num=outlier_num,
            exclude_process=exclude_process,
            use_case_branch=use_case_branch,
            metric_type=metric_type,
            min_row_count=min_row_count,
        )

        query = sql_template.get_iqr_outliers_template(
            first_groupby,
            second_groupby,
        )
        print(query)

        logger.info(f'SQL Query:\n'
                    f'{query}')

        self._execute_query(query, conn)
