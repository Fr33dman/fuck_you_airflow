from typing import Union, Dict

from .quantile_outlier_detector import QuantileOutlierDetector
from .iqr_outlier_detector import IQROutlierDetector
from .ensemble_outlier_detector import EnsembleOutlierDetector
from .outlier_detector_params import OutlierDetectorTypeEnum, \
    QuantileOutlierDetectorParams, IQROutlierDetectorParams, \
    EnsembleOutlierDetectorParams, DefaultOutlierDetectorParams


class OutlierDetectorFactory:
    """Factory for OutlierDetector"""
    @staticmethod
    def get_outlier_detector(
        outlier_detector_type: str,
        params: Union[DefaultOutlierDetectorParams, QuantileOutlierDetectorParams, IQROutlierDetectorParams, EnsembleOutlierDetectorParams]
    ):
        """
        Get specified outlier detector
        Parameters
        ----------
        outlier_detector_type : str
            Outlier detector type
        params : Union[QuantileOutlierDetectorParams, IQROutlierDetectorParams, EnsembleOutlierDetectorParams]
            Params for specified outlier detector
        Returns
        -------

        """
        if outlier_detector_type in (OutlierDetectorTypeEnum.DEFAULT, OutlierDetectorTypeEnum.QUANTILE):
            return QuantileOutlierDetector(params)
        elif outlier_detector_type == OutlierDetectorTypeEnum.IQR:
            return IQROutlierDetector(params)
        elif outlier_detector_type == OutlierDetectorTypeEnum.ENSEMBLE:
            return EnsembleOutlierDetector(params)
        else:
            raise ValueError(outlier_detector_type)
