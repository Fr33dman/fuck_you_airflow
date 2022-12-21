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
    table_schema='tableSchema',
    table_name='tableName',
    outlier_num=1,
    minimum_rows=40,
    sections=[
        'abc',
        'def'
    ],
    metric_type='metricType',
    use_case_branch=True,
    upper_bound=0.8,
    lower_bound=0.2,
    connection=conn,
)

manager.run()
