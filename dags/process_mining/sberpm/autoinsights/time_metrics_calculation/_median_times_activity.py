from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    Mapping,
    NamedTuple,
    TypeVar,
)

from numpy import float32, uint64, mean, median, trim_zeros

from process_mining.sberpm.autoinsights.time_metrics_calculation._activity_duration_super import (
    ActivityDurationSupervisor,
)
from process_mining.sberpm.autoinsights.time_metrics_calculation._duration_annotations import (
    TimestampActivity,
    ActivityDurationSeries,
    DurationsOfActivity,
)
from process_mining.sberpm.autoinsights.time_metrics_calculation._numpy_hash_codec import ArrayToHashSequenceCodec


class MeanDurationOfActivity(Generic[TypeVar("float32 | nan")]):  # TypeVar cause of bug for Union
    # nan for activity nodes with poor data - will be resolved in result
    pass


class MedianDurationOfActivity(Generic[TypeVar("float32 | nan")]):  # TypeVar cause of bug for Union
    # nan for activity nodes with poor data - will be resolved in result
    pass


class DurationOfActivityMeanMedianRatio(float32):
    pass


class ActivityMeanDurationSeries(Generic[TypeVar("Series", TimestampActivity, MeanDurationOfActivity)]):
    pass


class DurationsHashTable(Mapping[TimestampActivity, DurationsOfActivity]):
    pass


class MeanDurationsHashTable(Mapping[TimestampActivity, MeanDurationOfActivity]):
    pass


class MedianDurationsHashTable(Mapping[TimestampActivity, MeanDurationOfActivity]):
    pass


class AllMeanActivityDurations(NamedTuple):
    # intermediate RAM lightweight storage
    # if decreases efficiency, just slightly
    times: DurationsHashTable
    mean_times: MeanDurationsHashTable
    median_times: MedianDurationsHashTable


class ActivityDurationMeanMedianDifferential(
    Generic[TypeVar("Series", TimestampActivity, DurationOfActivityMeanMedianRatio)]
):
    # differential means here division of mean(timedelta) / median(timedelta)
    pass


@dataclass(init=False, eq=False, frozen=True)
class ActivityDurationMeanMedianInspector(ActivityDurationSupervisor):
    collection_of_activity_duration_modified: AllMeanActivityDurations = field(init=False)
    activity_duration_metrics: ActivityDurationMeanMedianDifferential = field(init=False, repr=False)

    _times_arrays_to_hash = ArrayToHashSequenceCodec(uint64)

    times_microdict_to_activity_data_times_dict: Dict = field(init=False, repr=False)
    mean_times_microdict_to_activity_data_mean_times_dict: Dict = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

    @property
    def get_collection_of_activity_duration_modified(
        self,
    ) -> AllMeanActivityDurations:
        activity_duration_mapping = self._hash_tables_mapping_from_activity_to_modified_duration

        return AllMeanActivityDurations(
            times=activity_duration_mapping.get("all_values")[0],
            mean_times=activity_duration_mapping.get("mean")[0],
            median_times=activity_duration_mapping.get("median")[0],
        )

    @property
    def get_times_microdict_to_activity_data_times_dict(self):
        activity_int_labels, activities = self._encoded_activities_map

        times_arrays_copy = self._times_arrays_to_hash.get_arrays.copy()
        duration_times = [trim_zeros(activity_times) / 1e6 for activity_times in times_arrays_copy]

        return dict(zip(activities, duration_times))

    @property
    def get_mean_times_microdict_to_activity_data_mean_times_dict(self):
        activity_int_labels, activities = self._encoded_activities_map
        mean_times_copy = self.collection_of_activity_duration_modified.mean_times.copy()

        for key in mean_times_copy:
            mean_times_copy[key] = mean_times_copy[key] / 1e6

        return dict(zip(activities, mean_times_copy.values()))

    @property
    def get_activity_duration_metrics(self) -> ActivityDurationMeanMedianDifferential:
        """
        Bisect rate of change of duration for each activity

        Returns:
            ActivityDurationMeanMedianDifferential: activities_duration_differential
        """
        return self._activities_duration_differential

    @property
    def _get_timedelta_modifiers(self) -> Dict[str, Callable]:
        def all_values(series):
            return series.values

        return dict(all_values=all_values, mean=mean, median=median)

    @property
    def _get_collection_of_activity_dividend(self) -> MeanDurationsHashTable:
        return self.collection_of_activity_duration_modified.mean_times

    @property
    def _get_collection_of_activity_divisor(self) -> MedianDurationsHashTable:
        return self.collection_of_activity_duration_modified.median_times

    @property
    def _yield_modified_variants_of_duration_data_by_modifier(
        self,
    ) -> Generator[ActivityDurationSeries, None, None]:
        yield from (
            {
                fn_name: self._retrieve_modified_activity_to_durations(self._activity_timestamp_data_slice, fn)
                for fn_name, fn in self._timedelta_modifiers.items()
            },
        )
