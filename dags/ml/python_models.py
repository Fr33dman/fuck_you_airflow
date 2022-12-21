import base64
from io import BytesIO
from typing import Dict
import json
import yaml

from ml.types import setup_dataset_params, setup_notation_params, \
    setup_research_params, get_columns_from_params, setup_model_params
from ml.universal_model_interface.python_models_utils import get_dataframe_from_python_model
from ml.universal_model_interface.data_types.types.front_response_type import FrontPythonModelResponse


meta_info_file = 'dags/ml/cfg/meta_info.yaml'
with open(meta_info_file, 'r') as file:
    meta_info = yaml.safe_load(file)


def calculate(data: Dict):
    ok = False
    result = None
    research_params = None
    dataset_params = None
    notation_params = None
    need_columns = None
    model_params = None
    filters_exist = False
    raw_model_params = None

    # Траим распарсить данные
    try:
        dataset_params = setup_dataset_params(data)
        research_params = setup_research_params(data)
        research_params.widget_type = 'python_models'
        notation_params = setup_notation_params(data)
        raw_model_params = FrontPythonModelResponse(**data.get("mlKeys"))
        model_params = setup_model_params(raw_model_params, meta_info)
        need_columns = get_columns_from_params(data, meta_info)
        filters_exist = False
        ok = True

    except Exception as err:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        result = json.dumps({
            'response': 'error',
            'message': str(err),
        })
    # Траим запустить и посчитать модель
    if ok:
        try:
            df = get_dataframe_from_python_model(
                research_params=research_params,
                dataset_params=dataset_params,
                notation_params=notation_params,
                python_model_params=model_params,
                columns_with_key=need_columns,
                filters=filters_exist,
            )
            result_file = BytesIO()
            df.to_excel(result_file, engine='xlsxwriter', index=False)
            result_file_byte2utf = base64.b64encode(result_file.getvalue()).decode('utf-8')
            result = json.dumps({
                    'name': raw_model_params.modelLabel,
                    'fileContent': result_file_byte2utf,
                })
        except MemoryError as err:
            import sys
            sys.exit('memory exceeded')

        except Exception as err:
            import sys, traceback
            traceback.print_exc(file=sys.stderr)

            result = json.dumps({
                'response': 'error',
                'message': str(err),
            })

    return result

