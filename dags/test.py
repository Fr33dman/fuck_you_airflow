from sqlalchemy.engine import create_engine
from requests import Session

from ml.new_outlier.outlier import OutlierManager


engine = create_engine(
            f'clickhouse+http://default:!QAZ1qaz'
            f'@10.53.222.170:8123/default',
            connect_args={'http_session': Session()}
        )

conn = engine.connect()


manager = OutlierManager(
    table_schema='_03ResearchData',
    table_name='td0002293_1452',
    outlier_num=3,
    minimum_rows=15,
    sections=[],
    metric_type='_activityDurationForward',
    use_case_branch=False,
    upper_bound=0.95,
    lower_bound=0.05,
    connection=conn,
)

manager.run()
