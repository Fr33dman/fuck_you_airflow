from typing import Generic, TypeVar, Dict, List

from numpy import float32, int64, int32


class BoolArray(Generic[TypeVar("array", bound=bool)]):
    pass


class FloatArray(Generic[TypeVar("array", bound=float32)]):
    pass


class LongLongArray(Generic[TypeVar("array", bound=int64)]):
    pass


class FloatSeries(Generic[TypeVar("Series", bound=float32)]):
    pass


class TimestampSeries(Generic[TypeVar("Series", bound=float32)]):
    pass


class FloatToLongMicrodict(Generic[TypeVar("mdict", int32, int64)]):
    pass


class TimestampActivity(str):
    pass


class ActivityIndex(Generic[TypeVar("Index", bound=TimestampActivity)]):
    pass


class DurationsOfActivity(FloatArray):
    pass


class ActivityDurationSeries(Generic[TypeVar("Series", TimestampActivity, DurationsOfActivity)]):
    pass


class ModifierName(str):
    pass


class ModifierMicrodictCollectionMapping(Dict[ModifierName, List[FloatToLongMicrodict]]):
    pass
