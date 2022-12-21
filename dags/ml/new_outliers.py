from typing import Dict
from requests import Session

from sqlalchemy.engine import create_engine

from ml.new_outlier.types.filter_request_dto import FilterRequestDTO
from ml.new_outlier.outlier import OutlierManager


def calculate(data: Dict):

    try:
        data = FilterRequestDTO(**data)
    except Exception as e:
        return

    engine = create_engine(
        f'clickhouse+http://default:!QAZ1qaz'
        f'@10.53.222.170:8123/default',
        connect_args={'http_session': Session()}
    )

    conn = engine.connect()

    manager = OutlierManager(
        table_schema=data.database,
        table_name=data.table_name,
        outlier_num=data.outlier_num,
        minimum_rows=data.min_row_count,
        sections=data.sections,
        metric_type=data.metric_type,
        use_case_branch=data.use_case_branch,
        upper_bound=data.outlier_detector_params.upper_bound,
        lower_bound=data.outlier_detector_params.lower_bound,
        connection=conn,
    )

    manager.run()
