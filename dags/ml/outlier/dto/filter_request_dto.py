from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, root_validator

from ml.outlier.outlier_detectors.outlier_detector_params import OutlierDetectorParams, \
    OutlierDetectorTypeEnum


DEFAULT_MIN_ROW_COUNT = 30


def strip_double_quotes(s: str) -> str:
    """
    Strip first and last double quotes
    Parameters
    ----------
    s : str
        Initial str
    Returns
    -------
    result_str : str
    """
    if s.startswith('"'):
        s = s[1:]
    if s.endswith('"'):
        s = s[:-1]

    return s


class FilterRequestDTO(BaseModel):
    database: str = Field(
        title='Название базы данных',
    )
    table_name: str = Field(
        title='Название таблицы из ClickHouse',
        alias='tableName',
    )
    metric_type: str = Field(
        title='Название поля для расчета',
        alias='metricType',
    )
    sections: List[Optional[str]] = Field(
        title='Список с названиями столбцов для разрезов',
        alias='sections',
        min_items=1,
        max_items=2,
    )
    outlier_detector_params: OutlierDetectorParams = Field(
        title='Параметры метода определения выбросов',
        alias='outlierDetectorParams',
    )
    min_row_count: int = Field(
        title='Минимальное число элементов для расчета выбросов',
        ge=0,
        alias='minRowCount',
    )
    exclude_process: bool = Field(
        title='Исключить процессы',
        alias='excludeProcess',
        default=False,
    )
    use_case_branch: bool = Field(
        title='Считать только метрику case',
        alias='useCaseBranch',
    )
    outlier_num: int = Field(
        title='Номер выбросов',
        alias='outlierNum',
    )

    class Config:
        allow_mutation = False

    @root_validator
    def validate_filter_request_dto(cls, values):
        sections = values.get('sections')
        print(values)

        if (values.get('outlier_detector_params').outlier_detector_type != OutlierDetectorTypeEnum.DEFAULT
                and len(sections) > 0 and sections[0] is None):
            raise ValueError(
                f'Expected first_groupby to be not None, but got None'
            )

        values['sections'] = list(map(lambda x: strip_double_quotes(x), sections))

        if values.get('outlier_detector_params').outlier_detector_type == OutlierDetectorTypeEnum.DEFAULT:
            values['min_row_count'] = DEFAULT_MIN_ROW_COUNT

        return values
