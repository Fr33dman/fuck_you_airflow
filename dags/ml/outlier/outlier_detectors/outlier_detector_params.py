from typing import Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class DefaultOutlierDetectorParams(BaseModel):
    lower_bound: float = Field(
        0.05,
        title='Уровень для нижней границы отсечения выбросов',
        const=True,
        alias='lowerBound',
    )
    upper_bound: float = Field(
        0.95,
        title='Уровень для верхней границы отсечения выбросов',
        const=True,
        alias='upperBound',
    )

    class Config:
        allow_mutation = False


class QuantileOutlierDetectorParams(BaseModel):
    lower_bound: float = Field(
        title='Уровень для нижней границы отсечения выбросов',
        ge=0.0,
        le=1.0,
        alias='lowerBound',
    )
    upper_bound: float = Field(
        title='Уровень для верхней границы отсечения выбросов',
        ge=0.0,
        le=1.0,
        alias='upperBound',
    )

    class Config:
        allow_mutation = False

    @validator('upper_bound')
    def check_if_upper_is_ge_lower(cls, v, values):
        if v < values['lower_bound']:
            raise ValueError(
                'upper_bound must be greater or equal than lower_bound'
            )

        return v


class IQROutlierDetectorParams(BaseModel):
    class Config:
        allow_mutation = False


class EnsembleOutlierDetectorParams(BaseModel):
    class Config:
        allow_mutation = False


class OutlierDetectorTypeEnum(str, Enum):
    DEFAULT = 'DEFAULT'
    QUANTILE = 'MANUAL_INPUT'
    IQR = 'STATIC_ALGO'
    ENSEMBLE = 'ML_ALGO'

    def __str__(self):
        return str(self.value)


class OutlierDetectorParams(BaseModel):
    outlier_detector_type: OutlierDetectorTypeEnum = Field(
        title='Тип определения выбросов',
        alias='outlierDetectorType',
    )
    params: Union[QuantileOutlierDetectorParams, DefaultOutlierDetectorParams,
                  IQROutlierDetectorParams, EnsembleOutlierDetectorParams] = Field(
        title='Параметры алгоритма определения выбросов'
    )

    class Config:
        allow_mutation = False

    @validator('params', pre=True)
    def check_params(cls, v, values):
        if values['outlier_detector_type'] == OutlierDetectorTypeEnum.DEFAULT:
            return DefaultOutlierDetectorParams()
        elif values['outlier_detector_type'] \
                == OutlierDetectorTypeEnum.QUANTILE:
            return QuantileOutlierDetectorParams(**v)
        elif values['outlier_detector_type'] == OutlierDetectorTypeEnum.IQR:
            return IQROutlierDetectorParams(**v)
        else:
            return EnsembleOutlierDetectorParams(**v)
