from typing import Optional, Union

from ml.outlier.outlier_detectors.outlier_detector_params import (
    DefaultOutlierDetectorParams,
    QuantileOutlierDetectorParams,
    IQROutlierDetectorParams,
    EnsembleOutlierDetectorParams
)


class OutlierSQLTemplate:
    """Base class for altering table before getting outlier's _caseId"""

    def __init__(
            self,
            table_schema: str,
            table_name: str,
            outlier_num: int,
            groups: list,
            use_case_branch: bool,
            metric_type: str,
            min_row_count: int,
            lower_bound: float,
            upper_bound: float,
    ):
        self.table_schema = table_schema
        self.table_name = table_name
        self.outlier_num = outlier_num
        self.use_case_branch = use_case_branch
        self.metric_type = metric_type
        self.min_row_count = min_row_count
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.first_groupby = groups[0] if len(groups) > 0 else None
        self.second_groupby = groups[1] if len(groups) > 1 else None

    def generate_resulting_table_creation_query(self) -> str:
        query = f'''
            CREATE TABLE {self.table_schema}.{self.table_name}outliers_{self.outlier_num} 
            (_caseId UInt64, _activityNum UInt16) ENGINE = MergeTree ORDER BY (_caseId, _activityNum);
        '''
        return query

    def generate_table_size_checking_query(self) -> str:
        table_postfix = 'c' if self.use_case_branch else 'a'
        query = f'''
            SELECT rows, round(bytes_on_disk / 1024/1024, 2) AS _size
            FROM system.parts
            WHERE database = '{self.table_schema}' AND table = '{self.table_name}{table_postfix}';
        '''
        return query

    def generate_quantiles_table_creation_query(self, quantile_table) -> str:
        query = f'''
            CREATE TABLE {self.table_schema}.{self.table_name}quantiles_{self.outlier_num}_{quantile_table}
            ENGINE = MergeTree ORDER BY _activityId AS
            SELECT
                _activityId,
                quantile({self.lower_bound})({self.metric_type}) AS lowerBound,
                quantile({self.upper_bound})({self.metric_type}) AS upperBound,
                count(DISTINCT _caseId) AS groupCount
                {f', `{self.first_groupby}`' if self.first_groupby else ''}
                {f', `{self.second_groupby}`' if self.second_groupby else ''}
            FROM {self.table_schema}.{self.table_name}a
            GROUP BY
                _activityId
                {f', `{self.first_groupby}`' if self.first_groupby else ''}
                {f', `{self.second_groupby}`' if self.second_groupby else ''};
        '''
        return query

    def generate_quantiles_table_drop_query(self, quantile_table) -> str:
        query = f'''
            DROP TABLE {self.table_schema}.{self.table_name}quantiles_{self.outlier_num}_{quantile_table}
        '''
        return query

    def generate_batch_case_quantiles_query(self, borders: list):
        case_borders = ', '.join(list(f'toString(quantileExact({border})(_caseId)) as case_{index}'
                                      for index, border in enumerate(borders)))
        query = f'''
            SELECT {case_borders}
            FROM {self.table_schema}.{self.table_name}a;
        '''
        return query

    def generate_calculation_outliers_batch_query(self, left_border, right_border, quantile_table) -> str:
        if left_border and right_border:
            border_condition = f'_caseId >= {left_border} AND _caseId < {right_border}'
        elif left_border:
            border_condition = f'_caseId >= {left_border}'
        else:
            border_condition = f'_caseId < {right_border}'

        query = f'''
        INSERT INTO {self.table_schema}.{self.table_name}outliers_{self.outlier_num}
        SELECT 
            _caseId, 
            _activityNum
        FROM (
            SELECT 
                _caseId, 
                _activityNum,
                if(
                    or(
                       {self.metric_type}
                       BETWEEN bounds.lowerBound
                       AND bounds.upperBound,
                       bounds.groupCount < {self.min_row_count}
                       ),
                    0,
                    1
                ) AS _isQuantileOutlier
            FROM {self.table_schema}.{self.table_name}a AS base_data
            
            JOIN (
                SELECT *
                FROM {self.table_schema}.{self.table_name}quantiles_{self.outlier_num}_{quantile_table}
            ) 
            AS bounds ON base_data._activityId = bounds._activityId
            {f'AND base_data.`{self.first_groupby}` = bounds.`{self.first_groupby}`' if self.first_groupby else ''}
            {f'AND base_data.`{self.second_groupby}` = bounds.`{self.second_groupby}`' if self.second_groupby else ''}
            
            WHERE {border_condition}
            )
        WHERE _isQuantileOutlier == 1
        GROUP BY
            _caseId,
            _activityNum;
        '''
        return query
