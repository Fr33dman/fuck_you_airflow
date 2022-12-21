from typing import Optional, Union

from ml.outlier.outlier_detectors.outlier_detector_params import (
    DefaultOutlierDetectorParams,
    QuantileOutlierDetectorParams,
    IQROutlierDetectorParams,
    EnsembleOutlierDetectorParams
)


class PrepareTableSQLTemplate:
    """Base class for altering table before getting outlier's _caseId"""

    def __init__(
            self,
            database: str,
            table_name: str,
            outlier_num: int,
    ):
        self.database = database
        self.table_name = table_name
        self.outlier_num = outlier_num

    def get_alter_table_template(self):
        database = self.database
        table_name = self.table_name
        outlier_num = self.outlier_num

        query = f"""
            CREATE TABLE IF NOT EXISTS {database}.{table_name}outliers_{outlier_num} 
            (_caseId UInt64, _activityNum UInt16) ENGINE = MergeTree ORDER BY (_caseId, _activityNum);
        """

        return query


class OutlierSQLTemplate:
    """Base class for generating SQL queries for outlier detectors"""

    def __init__(
            self,
            outlier_detector_params: Union[DefaultOutlierDetectorParams, QuantileOutlierDetectorParams,
                                           IQROutlierDetectorParams, EnsembleOutlierDetectorParams],
            database: str,
            table_name: str,
            outlier_num: int,
            exclude_process: bool,
            use_case_branch: bool,
            metric_type: str,
            min_row_count: int,
    ):
        self.params = outlier_detector_params
        self.database = database
        self.table_name = table_name
        self.exclude_process = exclude_process
        self.use_case_branch = use_case_branch
        self.metric_type = metric_type
        self.min_row_count = min_row_count
        self.outlier_num = outlier_num

    def get_quantile_outliers_template(self, first_groupby: str, second_groupby: Optional[str]):
        database = self.database
        table_name = self.table_name
        exclude_process = self.exclude_process
        use_case_branch = self.use_case_branch
        metric_type = self.metric_type
        min_row_count = self.min_row_count
        lower_bound = self.params.lower_bound
        upper_bound = self.params.upper_bound
        outlier_num = self.outlier_num

        activity_query = f'''
        INSERT INTO {database}.{table_name}outliers_{outlier_num}
        SELECT {'DISTINCT (_caseId), 0 as _activityNum' if exclude_process else '_caseId, _activityNum'}
        FROM
             (
              SELECT DISTINCT 
                              _caseId, 
                              _activityNum
              FROM
                   (
                    SELECT
                           base_data._caseId,
                           base_data._activityNum,
                           if(
                               or(
                                   base_data.{metric_type} 
                                   BETWEEN bounds._lowerBound
                                   AND bounds._upperBound,
                                   bounds._groupCount < {min_row_count}
                                   ),
                               0,
                               1
                               ) AS _isQuantileOutlier
                    FROM {database}.{table_name}a AS base_data
                        INNER JOIN (
                            SELECT
                                   _activityId,
                                   quantileExact({lower_bound})({metric_type}) AS _lowerBound,
                                   quantileExact({upper_bound})({metric_type}) AS _upperBound,
                                   count(*) AS _groupCount
                                   {f', `{first_groupby}`' if first_groupby else ''}
                                   {f', `{second_groupby}`' if second_groupby else ''}
                            FROM {database}.{table_name}a
                            GROUP BY
                            _activityId
                            {f', `{first_groupby}`' if first_groupby else ''}
                            {f', `{second_groupby}`' if second_groupby else ''}
                            ) AS bounds ON base_data._activityId = bounds._activityId
                                {f'AND base_data.`{first_groupby}` = bounds.`{first_groupby}`' if first_groupby else ''}
                                {f'AND base_data.`{second_groupby}` = bounds.`{second_groupby}`' if second_groupby else ''}
                   )
              WHERE _isQuantileOutlier == 1
              GROUP BY
                       _caseId, 
                       _activityNum
             );
          '''

        case_query = f'''
        INSERT INTO {database}.{table_name}outliers_{outlier_num}
        WITH
             cases AS (
                 SELECT DISTINCT
                                 _caseId,
                                 {metric_type},
                                 `{first_groupby}`
                                 {f', `{second_groupby}`' if second_groupby else ''}
                 FROM {database}.{table_name}a
                 ),
             bounds AS (
                 SELECT
                        quantileExact({lower_bound})({metric_type}) AS lowB,
                        quantileExact({upper_bound})({metric_type}) AS rigB,
                        count(distinct _caseId) AS countC,
                        `{first_groupby}`
                        {f', `{second_groupby}`' if second_groupby else ''}
                 FROM cases
                 GROUP BY
                          `{first_groupby}`
                          {f', `{second_groupby}`' if second_groupby else ''}
                 )
        SELECT _caseId as _caseId, 0 as _activityNum
        FROM (
            SELECT
                   cases._caseId,
                   if(
                       or(
                           {metric_type} BETWEEN lowB AND rigB,
                           countC < {min_row_count}
                           ),
                       0,
                       1
                       ) AS _isQuantileOutlier,
                   cases.`{first_groupby}`
                   {f', cases.`{second_groupby}`' if second_groupby else ''}
            FROM cases
            INNER JOIN
                bounds ON
                    cases.`{first_groupby}` = bounds.`{first_groupby}` 
                    {f'AND cases.`{second_groupby}` = bounds.`{second_groupby}`' if second_groupby else ''}
            )
        WHERE _isQuantileOutlier == 1
        GROUP BY _caseId;
        '''

        if use_case_branch:
            return case_query
        return activity_query

    def get_iqr_outliers_template(self, first_groupby: str, second_groupby: Optional[str]):
        database = self.database
        table_name = self.table_name
        exclude_process = self.exclude_process
        use_case_branch = self.use_case_branch
        metric_type = self.metric_type
        min_row_count = self.min_row_count
        outlier_num = self.outlier_num

        activity_query = f'''
        INSERT INTO {database}.{table_name}outliers_{outlier_num}
        SELECT {'DISTINCT(_caseId), 0 as _activityNum' if exclude_process else '_caseId, _activityNum'}
        FROM (
              SELECT
                     _caseId,
                     _activityNum
              FROM (
                    SELECT
                           base_data._caseId,
                           base_data._activityNum,
                           if(
                                   or(
                                       base_data.{metric_type}
                                       BETWEEN bounds._q1 - 1.5 * (bounds._q3 - bounds._q1)
                                       AND bounds._q3 + 1.5 * (bounds._q3 - bounds._q1),
                                       bounds._groupCount < {min_row_count}
                                       ),
                                   0,
                                   1
                               ) AS _isIQROutlier
        
                    FROM {database}.{table_name}a AS base_data
                        INNER JOIN (
                            SELECT
                                   _activityId,
                                   quantileExact(0.25)({metric_type}) AS _q1,
                                   quantileExact(0.75)({metric_type}) AS _q3,
                                   count(*)                           AS _groupCount,
                                   `{first_groupby}`
                                   {f', `{second_groupby}`' if second_groupby else ''}
        
                            FROM {database}.{table_name}a
                            GROUP BY
                                     _activityId,
                                     `{first_groupby}`
                                     {f', `{second_groupby}`' if second_groupby else ''}
                                     ) AS bounds ON base_data._activityId = bounds._activityId
                                         AND base_data.`{first_groupby}` = bounds.`{first_groupby}`
                                         {f'AND base_data.`{second_groupby}` = bounds.`{second_groupby}`' if second_groupby else ''}
                   )
              WHERE _isIQROutlier == 1
              GROUP BY
                       _caseId,
                       _activityNum
         );
        '''

        case_query = f'''
        INSERT INTO {database}.{table_name}outliers_{outlier_num}
        WITH
             cases AS (
                 SELECT DISTINCT
                                 _caseId,
                                 {metric_type},
                                 `{first_groupby}`
                                 {f', `{second_groupby}`' if second_groupby else ''}
                 FROM {database}.{table_name}a
                 ),
             bounds AS (
                 SELECT
                        quantileExact(0.25)({metric_type}) AS lowB,
                        quantileExact(0.75)({metric_type}) AS rigB,
                        count(distinct _caseId) AS countC,
                        `{first_groupby}`
                        {f', `{second_groupby}`' if second_groupby else ''}
                 FROM cases
                 GROUP BY
                          `{first_groupby}`
                          {f', `{second_groupby}`' if second_groupby else ''}
                 )
        SELECT _caseId as _caseId, 0 as _activityNum
        FROM (
            SELECT
                   cases._caseId,
                   if(
                       or(
                           {metric_type} BETWEEN bounds.lowB - 1.5 * (bounds.rigB - bounds.lowB)
                               AND bounds.rigB + 1.5 * (bounds.rigB - bounds.lowB),
                           countC < {min_row_count}
                           ),
                       0,
                       1
                       ) AS _isQuantileOutlier,
                   cases.`{first_groupby}`
                   {f', cases.`{second_groupby}`' if second_groupby else ''}
            FROM cases
            INNER JOIN
                bounds ON
                    cases.`{first_groupby}` = bounds.`{first_groupby}` 
                    {f'AND cases.`{second_groupby}` = bounds.`{second_groupby}`' if second_groupby else ''}
            )
        WHERE _isQuantileOutlier == 1
        GROUP BY _caseId;
        '''

        if use_case_branch:
            return case_query
        return activity_query

    def get_ensemble_outliers_load_data_template(self, first_groupby: str, second_groupby: Optional[str]):
        database = self.database
        table_name = self.table_name
        metric_type = self.metric_type
        first_groupby = first_groupby
        second_groupby = second_groupby
        use_case_branch = self.use_case_branch

        activity_query = f"""
        SELECT 
            _caseId,
            _activityNum, 
            {metric_type}
        FROM 
          {database}.{table_name}a 
        GROUP BY 
            _caseId, 
            _activityNum,
            {metric_type},
            `{first_groupby}`
            {f', `{second_groupby}`' if second_groupby else ''} 
        """

        case_query = f'''
        SELECT DISTINCT 
            _caseId,
            0 as _activityNum, 
            {metric_type}
        FROM 
          {database}.{table_name}a
        GROUP BY 
            _caseId,
            {metric_type},
            `{first_groupby}`
            {f', `{second_groupby}`' if second_groupby else ''}
        '''

        if use_case_branch:
            return case_query
        return activity_query
