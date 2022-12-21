import json
from typing import Dict
from sqlalchemy import create_engine
from requests import Session

from ml.outlier.dto.filter_request_dto import FilterRequestDTO
from ml.outlier.dto.filter_response_dto import ContentResponseDTO
from ml.outlier.outlier_detectors.outlier_detector_factory import OutlierDetectorFactory


def calculate(data: Dict):

    ok = False
    content = None

    try:
        data = FilterRequestDTO(**data)
        ok = True

    except Exception as e:
        import traceback, sys
        err = str(e)
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        message_error = f"An error occurred on line {line} " \
                         f"in file {filename} in statement {text}, " \
                         f"error - {err}"

        outlierNum = data.get('outlierNum', None)
        content = json.dumps(ContentResponseDTO(
            response='error',
            message=message_error,
            outlierNum=outlierNum,
        ).dict())

    if ok:
        outlier_detector = OutlierDetectorFactory.get_outlier_detector(
            outlier_detector_type=data.outlier_detector_params.outlier_detector_type,
            params=data.outlier_detector_params.params
        )

        engine = create_engine(
            f'clickhouse+http://default:!QAZ1qaz'
            f'@10.53.222.170:8123/default',
            connect_args={'http_session': Session()}
        )

        conn = engine.connect()

        try:

            outlier_detector.apply(
                database=data.database,
                table_name=data.table_name,
                outlier_num=data.outlier_num,
                exclude_process=False,  # Всегда считаем по case + activity
                use_case_branch=data.use_case_branch,
                first_groupby=data.sections[0],
                second_groupby=data.sections[1] if len(data.sections) > 1 else None,
                metric_type=data.metric_type,
                min_row_count=data.min_row_count,
                conn=conn,
            )

            conn.close()

        except Exception as e:
            import traceback, sys
            err = str(e)
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            message_error = f"An error occurred on line {line} " \
                            f"in file {filename} in statement {text}, " \
                            f"error - {err}"

            content = json.dumps(ContentResponseDTO(
                response='error',
                message=message_error,
                outlierNum=data.outlier_num,
            ).dict())

        else:

            content = json.dumps(ContentResponseDTO(
                response='ok',
                outlierNum=data.outlier_num,
            ).dict())

    return content
