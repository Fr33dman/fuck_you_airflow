import datetime as dt
from requests import Session
from sqlalchemy.engine import create_engine

from airflow.models import DAG
from airflow.operators.python import PythonOperator

from ml.new_outlier.types.filter_request_dto import FilterRequestDTO
from ml.new_outlier.outlier import OutlierManager

args = {
    'owner': 'airflow',
    'start_date': dt.datetime.utcnow(),
    'provide_context': True,
    'retries': 5,
}


def new_outliers(**kwargs):
    config = kwargs['dag_run'].conf
    print(f'\n\n{config}\n\n')
    if 'conf' in config:
        config = config.get('conf')
    data = config.get('outliers')

    data = FilterRequestDTO(**data)
    print(data.dict())

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
    return 1


with DAG(dag_id='test_outliers_batch', default_args=args, schedule_interval=None) as dag:
    outlier = PythonOperator(
        task_id='new_outliers',
        python_callable=new_outliers,
        provide_context=True,
        dag=dag,
    )
