from abc import ABC, abstractmethod
from typing import Optional

from clickhouse_driver.dbapi.connection import Connection


class BaseOutlierDetector(ABC):
    """Base abstract class for OutlierDetector"""
    @abstractmethod
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
        """
        Apply will result in updating _caseId in outliersC table in ClickHouse
        Parameters
        ----------
        table_name : str
            Name of the table to apply filtering for
        database : str
            Name of database where table is
        outlier_num: int
            Number of outlier (user has some outliers, that saves in several tables)
        exclude_process: bool
            Count cases only or cases with activities
        use_case_branch: bool
            Use table C and calculate only case without activities
        first_groupby : str
            Name of the first groupby
        second_groupby : Optional[str]
            Name of the second groupby
        metric_type : str
            Metric name
        min_row_count : int
            Minimal amount of rows to calculate outliers
        conn : Connection
            DBAPI connection to ClickHouse
        Returns
        -------
        """
        pass
