from numbers import Number
from typing import Optional, List, Tuple
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sqlalchemy.engine import Connection

from ml.outlier.outlier_detectors.base_outlier_detector import BaseOutlierDetector
from ml.outlier.outlier_detectors.outlier_detector_params import EnsembleOutlierDetectorParams
from ml.outlier.outlier_detectors.sql_templates import OutlierSQLTemplate, \
    PrepareTableSQLTemplate


class EnsembleOutlierDetector(BaseOutlierDetector):
    """Ensemble (result from disjunction of DBSCAN and One-class SVM)"""
    SVM_PARAMS = {
        "cache_size": 200,
        "coef0": 0.0,
        "degree": 4,
        "gamma": "scale",
        "kernel": "poly",
        "max_iter": -1,
        "nu": 0.3,
        "shrinking": True,
        "tol": 0.001,
        "verbose": False
    }
    DBSCAN_PARAMS = {
        "algorithm": "auto",
        "eps": 0.1,
        "leaf_size": 30,
        "metric": "euclidean",
        "metric_params": None,
        "min_samples": 9,
        "n_jobs": None,
        "p": None
    }

    def __init__(self, params: EnsembleOutlierDetectorParams):
        self.params = params

    @staticmethod
    def _fetch_data_from_clickhouse(query: str, conn: Connection) -> List[Tuple[Number, Number, Number]]:
        """
        Fetch data from ClickHouse with particular SQL query
        Parameters
        ----------
        query : str
            SQL query
        conn : Connection
            DBAPI connection
        Returns
        -------
        data : List[Tuple[Number, Number, Number]]
        """
        data = conn.execute(query).fetchall()
        return data

    @staticmethod
    def _insert_data_to_clickhouse(
        data: List[Tuple[int, int]],
        database: str,
        table_name: str,
        outlier_num: int,
        conn: Connection
    ):
        """
        Insert data to ClickHouse
        Parameters
        ----------
        data : List[Tuple[int, str, Number]]
        database : str
        table_name : str
        conn : Connection

        Returns
        -------
        """
        if len(data) == 0:
            return

        rows = ''.join(list(f'({row[0]}, {row[1]}), ' for row in data))
        conn.execute(f'INSERT INTO {database}.{table_name}outliers_{outlier_num} VALUES {rows}')

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
        alter_table_query = PrepareTableSQLTemplate(
            database,
            table_name,
            outlier_num,
        ).get_alter_table_template()
        conn.execute(alter_table_query)

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
        query = sql_template.get_ensemble_outliers_load_data_template(
            first_groupby,
            second_groupby
        )

        data = self._fetch_data_from_clickhouse(query, conn)
        outliers_set = set()

        case_ids, activity_nums, durations = zip(*data)
        outliers_binary_mask = self._get_outliers(
            np.array(durations).reshape(-1, 1)
        )

        for label, case_id, activity_num in zip(outliers_binary_mask, case_ids, activity_nums):
            if label:
                if exclude_process:
                    outliers_set.add((case_id, 0))
                else:
                    outliers_set.add((case_id, activity_num))

        self._insert_data_to_clickhouse(
            data=list(outliers_set),
            database=database,
            table_name=table_name,
            outlier_num=outlier_num,
            conn=conn
        )

    def _get_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Return binary mask: is object outlier or not
        Parameters
        ----------
        data : np.ndarray
            NumPy array with shape [len, 1]

        Returns
        -------
        indices : np.ndarray
            Binary mask of outliers
        """
        if len(data) == 0:
            return np.array([])

        ocsvm: OneClassSVM = OneClassSVM().set_params(**self.SVM_PARAMS)
        dbscan: DBSCAN = DBSCAN().set_params(**self.DBSCAN_PARAMS)

        ocsvm_predictions = ocsvm.fit_predict(data)
        dbscan_predictions = dbscan.fit_predict(data)

        ocsvm_labels = np.where(
            ocsvm_predictions == -1,
            True,
            False
        )
        dbscan_labels = np.where(
            dbscan_predictions == -1,
            True,
            False
        )

        disjunction_predictions = np.logical_or(
            ocsvm_labels,
            dbscan_labels
        )

        return disjunction_predictions
