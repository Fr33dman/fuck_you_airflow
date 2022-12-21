import datetime as dt

from airflow.models import DAG
from airflow.operators.python import PythonOperator

from ml.python_models import calculate as python_models_calculate
from ml.outliers import calculate as outlier_calculate


args = {
    'owner': 'airflow',
    'start_date': dt.datetime.utcnow(),
    'provide_context': True,
    'retries': 5,
}


def run_python_models(**kwargs):
    config = kwargs['dag_run'].conf
    data = config.get('python_models')
    try:
        python_models_calculate(data)
    except Exception as e:
        print(e)
        print(e.with_traceback())
    return 1


def outliers(**kwargs):
    config = kwargs['dag_run'].conf
    data = config.get('outliers')
    
    outlier_calculate(data)
    return 1


with DAG(dag_id='test_pm_pipeline', default_args=args, schedule_interval=None) as dag:
    python_model = PythonOperator(
        task_id='python_models',
        python_callable=run_python_models,
        provide_context=True,
        dag=dag,
    )
    outlier = PythonOperator(
        task_id='outliers',
        python_callable=outliers,
        provide_context=True,
        dag=dag,
    )
