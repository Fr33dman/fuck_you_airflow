from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    Mapping,
    NamedTuple,
    Tuple,
    TypeVar,
)

from numpy import float32, mean, roll
from numpy import ptp as min_max_interval
from pandas import DataFrame

from process_mining.sberpm.autoinsights.time_metrics_calculation._activity_duration_super import ActivityDurationSupervisor
from process_mining.sberpm.autoinsights.time_metrics_calculation._duration_annotations import (
    BoolArray,
    TimestampActivity,
    ActivityDurationSeries,
)


class MeanDurationOfActivity(Generic[TypeVar("float32 | nan")]):  # TypeVar cause of bug for Union
    # nan for activity nodes with poor data - will be resolved in result
    pass


class DurationOfActivityBisectRatio(float32):
    pass


class ActivityMeanDurationSeries(Generic[TypeVar("Series", TimestampActivity, MeanDurationOfActivity)]):
    pass


class MeanDurationsHashTable(Mapping[TimestampActivity, MeanDurationOfActivity]):
    pass


class BisectedLogActivityDurations(NamedTuple):
    # intermediate RAM lightweight storage
    # if decreases efficiency, just slightly
    first_half_mean_times: MeanDurationsHashTable
    second_half_mean_times: MeanDurationsHashTable


class ActivityDurationBisectDifferential(
    Generic[TypeVar("Series", TimestampActivity, DurationOfActivityBisectRatio)]
):
    # differential means here division of (end_second - start_second) / (end_first - start_first)
    pass


@dataclass(init=False, eq=False, frozen=True)
class ActivityDurationChangeInspector(ActivityDurationSupervisor):
    collection_of_activity_duration_modified: BisectedLogActivityDurations = field(init=False)
    activity_duration_metrics: ActivityDurationBisectDifferential = field(init=False, repr=False)

    _data_middle_timestamp: float32 = field(init=False)

    __data_frames_from_both_sides_of_middle_time: Tuple[DataFrame, DataFrame] = field(init=False, repr=False)
    __middle_time_data_mask: BoolArray = field(init=False, repr=False)
    __mask_for_both_timestamps: BoolArray = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

    @property
    def get_collection_of_activity_duration_modified(self) -> BisectedLogActivityDurations:
        return BisectedLogActivityDurations(*self._hash_tables_mapping_from_activity_to_modified_duration["mean"])

    @property
    def get_activity_duration_metrics(self) -> ActivityDurationBisectDifferential:
        """
        Bisect rate of change of duration for each activity

        Returns:
            ActivityDurationBisectDifferential: activities_duration_differential
        """
        return self._activities_duration_differential

    @property
    def _get_timedelta_modifiers(self) -> Dict[str, Callable]:
        return dict(mean=mean)

    @property
    def _get_collection_of_activity_dividend(self) -> MeanDurationsHashTable:
        return self.collection_of_activity_duration_modified.second_half_mean_times

    @property
    def _get_collection_of_activity_divisor(self) -> MeanDurationsHashTable:
        return self.collection_of_activity_duration_modified.first_half_mean_times

    @property
    def _yield_modified_variants_of_duration_data_by_modifier(
        self,
    ) -> Generator[ActivityDurationSeries, None, None]:
        yield from (
            {
                fn_name: self._retrieve_modified_activity_to_durations(activity_timestamp_data_half, fn).astype(
                    float32
                )  # buggy thing might change dtype
                for fn_name, fn in self._timedelta_modifiers.items()
            }
            for activity_timestamp_data_half in self.__data_frames_from_both_sides_of_middle_time
        )

    @property
    def _get_data_middle_timestamp(self) -> float32:
        first_timestamp_series = self._activity_timestamp_data_slice[self._timestamp_columns].iloc[
            :, 0  # get first datetime series - most likely start timestamp
        ]

        return first_timestamp_series.min() + min_max_interval(first_timestamp_series) / 2

    def __request_mask_for_only_timestamp(self, timestamp_name, timestamp_shift) -> BoolArray:
        return (
            roll(
                self._activity_timestamp_data_slice.copy()[
                    timestamp_name
                ],  # use copy as roll changes data inplace
                shift=timestamp_shift,
            )
            <= self._data_middle_timestamp
        )

    @property
    def _get_mask_for_both_timestamps(self) -> BoolArray:
        return (
            self._activity_timestamp_data_slice[self._start_timestamp_name] <= self._data_middle_timestamp
        ).to_numpy()

    @property
    def _get_middle_time_data_mask(self) -> BoolArray:
        """
        Get a boolean mask that is True if the data timestamp is lower than the middle time
        """
        # timestamp roll is required for data with only timestamp
        if not self._end_timestamp_name:
            return self.__request_mask_for_only_timestamp(self._start_timestamp_name, timestamp_shift=-1)
        if not self._start_timestamp_name:
            return self.__request_mask_for_only_timestamp(self._end_timestamp_name, timestamp_shift=1)
        # both timestamps - no need to roll
        return self.__mask_for_both_timestamps

    @property
    def _get_data_frames_from_both_sides_of_middle_time(
        self,
    ) -> Tuple[DataFrame, DataFrame]:
        timestamp_mask = self.__middle_time_data_mask  # timestamp split edge

        return (
            self._activity_timestamp_data_slice[timestamp_mask],
            self._activity_timestamp_data_slice[~timestamp_mask],
        )
