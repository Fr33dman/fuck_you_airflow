from abc import abstractmethod, ABC
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import zip_longest
from re import sub
from typing import (
    Callable,
    Deque,
    Generator,
    Literal,
    Optional,
    Tuple,
    Union,
    Dict,
)

from microdict import mdict  # memory eff hash table, faster than dict

from numpy import float64, float32, int32, int64, uint64, isnan, nan, stack, argmax, pad, inf
from numpy import roll
from pandas import Categorical, DataFrame, Index, Series, Timestamp, factorize
from pandas.core.groupby.generic import DataFrameGroupBy

from loguru import logger

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.autoinsights.time_metrics_calculation._duration_annotations import (
    BoolArray,
    LongLongArray,
    FloatSeries,
    TimestampSeries,
    ActivityIndex,
    ActivityDurationSeries,
    ModifierMicrodictCollectionMapping,
)


class ModifiedDurationOfActivity(Enum):
    MeanDurationOfActivity = auto()
    MedianDurationOfActivity = auto()


class ActivityModifiedDurationSeries(Enum):
    ActivityDurationSeries = auto()
    ActivityMeanDurationSeries = auto()


class ModifiedDurationsHashTable(Enum):
    MeanDurationsHashTable = auto()
    MedianDurationsHashTable = auto()


class ModifiedDurationsCollection(Enum):
    BisectedLogActivityDurations = auto()
    AllMeanActivityDurations = auto()


class ActivityDurationModifiedDifferential(Enum):
    ActivityDurationBisectDifferential = auto()
    ActivityDurationMeanMedianDifferential = auto()


@dataclass(eq=False, frozen=True)
class ActivityDurationSupervisor(ABC):
    data_holder: DataHolder

    _activities_duration_differential: ActivityDurationModifiedDifferential = field(init=False, repr=False)
    _collection_of_activity_dividend: ModifiedDurationsHashTable = field(init=False, repr=False)
    _collection_of_activity_divisor: ModifiedDurationsHashTable = field(init=False, repr=False)
    _hash_tables_mapping_from_activity_to_modified_duration: ModifierMicrodictCollectionMapping = field(
        init=False, repr=False
    )

    _activity_timestamp_data_slice: DataFrame = field(init=False, repr=False)
    _timestamp_columns: Index = field(init=False)
    _timedelta_modifiers: Dict[str, Callable] = field(init=False)
    _id_name: str = field(init=False)
    _activity_name: str = field(init=False)
    _start_timestamp_name: Optional[str] = field(init=False)
    _end_timestamp_name: Optional[str] = field(init=False)
    _encoded_activities_map: Tuple[LongLongArray, ActivityIndex] = field(init=False, repr=False)

    __first_yield_data_len_queue: Deque = field(default_factory=lambda: deque(), repr=False)
    __timedelta_series_on_timestamp_kind: FloatSeries = field(init=False, repr=False)
    __durations_diff_ratio_on_activity: ActivityDurationModifiedDifferential = field(init=False, repr=False)

    def __post_init__(self):
        self.__convert_datetime_to_timestamp()
        self.__convert_activity_to_categorical()

    def __getattr__(self, name):
        """
        Lazily get attributes of any dataclass through property on attribute access only.
        Allows to not to enumerate properties like self.attr = self._get_attr in __post_init__,
            will call property for attribute one time, then will just get the value.
        Definition of attributes through @property must comply '_get_<attr_name>' convention,
            otherwise it will lead to an error.
        If there is no '_get_<attr_name>' property in self,
            it will search for '_get_<attr_name>' property among parent classes.

        Args:
            name str: name of the attribute

        Raises:
            AttributeError: Error informing that accessed attribute is not present in the dataclass

        Returns: class attribute with name passed through <instance>.<name> call
        """
        if name not in self.__dataclass_fields__.keys():
            raise AttributeError(f"{self} has no attribute called {name}")

        if not name.startswith("_"):  # is public
            property_get_expression = f"self.get_{name}"
        else:
            no_privacy_name = sub(
                rf"^_({super().__thisclass__.__name__}|{super().__self_class__.__name__})?(__)?",
                "",
                name,
            )

            property_get_expression = f"self._get_{no_privacy_name}"

        object.__setattr__(
            self,
            name,
            eval(property_get_expression),
        )  # set attr with property of _?get_<attr_name>

        return getattr(self, name)

    @abstractmethod
    def get_collection_of_activity_duration_modified(self) -> ModifiedDurationsCollection:
        """
        Use mdict collections to effectively store and then return all of metrics like mean_median, mean_time, time
        """

    @abstractmethod
    def get_activity_duration_metrics(self) -> ActivityDurationModifiedDifferential:
        """
        Abstract public method to return activity duration metrics
        depending on data and metrics kind required
        """

    @abstractmethod
    def _get_timedelta_modifiers(self) -> Dict[str, Callable]:
        pass

    @abstractmethod
    def _get_collection_of_activity_dividend(self) -> ModifiedDurationsHashTable:
        pass

    @abstractmethod
    def _get_collection_of_activity_divisor(self) -> ModifiedDurationsHashTable:
        pass

    @abstractmethod
    def _yield_modified_variants_of_duration_data_by_modifier(
        self,
    ) -> Generator[ActivityDurationSeries, None, None]:
        pass

    # * TODO rm copy once transfer to DataHolder methods done
    @property
    def _get_activity_timestamp_data_slice(self):
        return self.data_holder.data[[self._id_name, self._activity_name, *self._timestamp_columns]]

    @property
    def _get_id_name(self) -> str:
        return self.data_holder.id_column

    @property
    def _get_activity_name(self) -> str:
        return self.data_holder.activity_column

    @property
    def _get_start_timestamp_name(self) -> str:
        return self.data_holder.start_timestamp_column

    @property
    def _get_end_timestamp_name(self) -> str:
        return self.data_holder.end_timestamp_column

    # * TODO __get_timestamp_columns DataHolderSorter -> transfer DataHolder
    @property
    def _get_timestamp_columns(self) -> Index:
        return self.data_holder.data.select_dtypes(
            ["datetime", "datetimetz"],
        ).columns  # start_timestamp, end_timestamp

    @property
    def _get_activities_duration_differential(self) -> ActivityDurationModifiedDifferential:
        """
        Calculate ratio of mean time being in node from later times data half
            to node from earlier times data half
        Ratio is calculated for each activity
        """
        activity_durations_diff = self.__durations_diff_ratio_on_activity.replace(
            inf,
            nan,
        )  # replace inf with nan values -> drop and show warning

        self.__warn_calculated_unsuccessfully_for_activity_diff(activity_durations_diff)

        return activity_durations_diff.fillna(
            1,  # set activity duration change to 1 for unsuccessful activities
        ).to_dict()

    @property
    def _get_hash_tables_mapping_from_activity_to_modified_duration(
        self,
    ) -> ModifierMicrodictCollectionMapping:
        activity_duration_mappings_by_modifiers = {fn_name: [] for fn_name in self._timedelta_modifiers}
        modified_duration_data_gen = self._yield_modified_variants_of_duration_data_by_modifier

        for activity_to_duration_mapping in modified_duration_data_gen:
            hash_table_gen = self.__yield_hash_table_of_modified_durations

            for _, duration_modifier in zip(
                hash_table_gen, activity_duration_mappings_by_modifiers
            ):  # zip of modifier keys, steps of modified duration vector receiving generator
                activity_to_duration: ActivityModifiedDurationSeries = activity_to_duration_mapping[
                    duration_modifier
                ]

                durations_hash_table = hash_table_gen.send(activity_to_duration)
                activity_duration_mappings_by_modifiers[duration_modifier].append(durations_hash_table)

        return activity_duration_mappings_by_modifiers

    def _retrieve_modified_activity_to_durations(
        self, activity_timestamp_df: DataFrame, modifier: Callable
    ) -> ActivityDurationSeries:
        # just one-time save len of yielded data
        self.__first_yield_data_len_queue.append(len(activity_timestamp_df))

        return self.__request_function_apply_to_grouped(
            self.__retrieve_modified_timedelta_for_activity_series,
            activity_timestamp_df,
            modifier,
        )

    # * TODO transfer to DataHolder
    def __convert_datetime_to_timestamp(self) -> None:
        def pandas_timestamp_to_float(datetime: Timestamp):
            return datetime.timestamp()

        for timestamp_name in self._timestamp_columns:
            self._activity_timestamp_data_slice[timestamp_name] = (
                self._activity_timestamp_data_slice[timestamp_name]
                .map(pandas_timestamp_to_float)
                .astype(float64, copy=False)  # use 64 to operate with microseconds
            )

    # * TODO transfer to DataHolder
    def __convert_activity_to_categorical(self) -> None:
        self._activity_timestamp_data_slice[self._activity_name] = Categorical(
            self._activity_timestamp_data_slice[self._activity_name]
        )

    def __collect_dt_series_and_id_duplicates_mask(
        self,
        data: DataFrame,
        dt_column_name: Literal["start_timestamp_column", "end_timestamp_column"],
    ) -> Tuple[TimestampSeries, BoolArray]:
        return (data[dt_column_name], data.duplicated(self._id_name).values)

    def __calculate_delta_seconds_via_datetime_column_diff(
        self,
        timestamp_series: TimestampSeries,
        duplicated_id_values: BoolArray,
    ) -> FloatSeries:
        """
        Calculate mean difference between timestamps of the same id in the data sorted by id

        Args:
            DataFrame: dataframe with at least case_id, activity and timestamp sorted by case_id

        Returns:
            Series: activity durations in seconds of same ids
        """
        return timestamp_series.diff(  # difference between following timestamp rows
            periods=1,  # ascending, no gap
        ).where(
            duplicated_id_values,
            nan,
        )  # bool mask: current id value duplicates previous

    @property
    def _get_timedelta_series_on_timestamp_kind(self) -> FloatSeries:
        # popped first part of self._activity_timestamp_data_slice
        # from first data value from _yield_modified_variants_of_duration_data_by_modifier
        len_of_yielded_activity_timestamp_data = self.__first_yield_data_len_queue.pop()

        if not self._end_timestamp_name:
            shift = -1 if len_of_yielded_activity_timestamp_data < len(self._activity_timestamp_data_slice) else 0

            return Series(
                roll(
                    self.__calculate_delta_seconds_via_datetime_column_diff(
                        *self.__collect_dt_series_and_id_duplicates_mask(
                            self._activity_timestamp_data_slice.copy(),  # use copy as roll changes data inplace
                            self._start_timestamp_name,  # diff on start_timestamp
                        )
                    ),  # get delta seconds (for activities) by idea of subtracting following values (sorted) of same process
                    shift=shift,
                ),  # shift datetime series backwards
                index=self._activity_timestamp_data_slice.index,  # preserve index of data
            )

        if not self._start_timestamp_name:
            return self.__calculate_delta_seconds_via_datetime_column_diff(
                *self.__collect_dt_series_and_id_duplicates_mask(
                    self._activity_timestamp_data_slice,
                    self._end_timestamp_name,  # diff on end_timestamp
                )
            )  # get delta seconds (for activities) by idea of subtracting following values (sorted) of same process

        # both timestamps - ignore diff for start-end datetime interval - calculate (end - start) timedelta later
        return Series(dtype=float64)

    def __request_data_grouped_by_activity(self, data: DataFrame) -> DataFrameGroupBy:
        return data.groupby(
            self.data_holder.activity_column,
        )

    def __shift_floating_point_microseconds_right(self, duration: float) -> float:
        float_precision_multiplier = 1e6

        return duration * float_precision_multiplier

    def __retrieve_zero_padded_matrix_or_array(self, input_array):
        """
        Method similar to pad_sequence for both of numpy array and array of arrays

        Arguments:
            input_array: numpy array

        Returns:
            array: Same array for 1D array and 2D array from nested
        """
        maxsize = input_array[
            argmax(
                [*map(lambda arr: arr.size, input_array)],
            )
        ].size

        def pad_tail_zero(inner_array, pad_end=maxsize):
            return pad(inner_array, pad_width=(0, (pad_end - inner_array.size)), constant_values=0)

        return stack(
            [
                *map(
                    pad_tail_zero,
                    input_array,
                )
            ],
        ).astype(uint64)

    def __request_function_apply_to_grouped(
        self, fn: Callable, data: DataFrame, timedelta_modifier: Callable
    ) -> DataFrame:
        def get_apply_function_with_modifier(data_frame):
            return fn(data_frame, timedelta_modifier)

        return self.__retrieve_zero_padded_matrix_or_array(
            self.__shift_floating_point_microseconds_right(
                self.__request_data_grouped_by_activity(data).apply(get_apply_function_with_modifier).to_numpy()
            ),
        )

    def __retrieve_modified_timedelta_for_activity_series(
        self, activity_timestamp_data: DataFrame, modifier: Callable
    ) -> ModifiedDurationOfActivity:
        if not (no_end_or_start_timedelta_series := self.__timedelta_series_on_timestamp_kind).empty:
            timedelta_for_activity = no_end_or_start_timedelta_series[activity_timestamp_data.index]

            return modifier(
                timedelta_for_activity[
                    ~isnan(
                        timedelta_for_activity,
                    )  # some ids timedelta cannot be calculated with single timestamp feature
                ]
            )

        # if has both start and end timestamps
        timedelta_seconds_series = (
            activity_timestamp_data[self._end_timestamp_name] - activity_timestamp_data[self._start_timestamp_name]
        )

        return modifier(timedelta_seconds_series[~isnan(timedelta_seconds_series)])

    @property
    def _get_encoded_activities_map(self) -> Tuple[LongLongArray, ActivityIndex]:
        return factorize(self._activity_timestamp_data_slice[self._activity_name].values.categories)

    @property
    def __yield_hash_table_of_modified_durations(
        self,
    ) -> Generator[
        ModifiedDurationsHashTable, Union[ActivityModifiedDurationSeries, ActivityDurationSeries], None
    ]:
        while True:
            activity_to_duration_values = yield

            activity_int_labels, _ = self._encoded_activities_map

            if len(activity_to_duration_values.shape) > 1:
                self._times_arrays_to_hash.encode_arrays(activity_to_duration_values)
                activity_to_duration_values = self._times_arrays_to_hash.get_array_hashes  # int64

            durations_hash_table = mdict.create("i32:i64")

            for activity, activity_durations_modified in zip(
                activity_int_labels.astype(int32, copy=False),
                activity_to_duration_values.astype(int64, copy=False),
                # activity_to_duration_long_values.astype(int64, copy=False),
            ):
                durations_hash_table[activity] = activity_durations_modified

            yield durations_hash_table

    @property
    def _get_durations_diff_ratio_on_activity(self) -> ActivityDurationModifiedDifferential:
        def microdict_to_activity_series(mdict_: mdict) -> ActivityDurationSeries:
            _, all_activities = self._encoded_activities_map

            return Series(
                DataFrame(
                    [
                        *zip_longest(
                            all_activities,
                            mdict_.values(),
                        ),  # some of mean duration values may be missing
                    ]
                ).set_index(
                    0,
                )[  # set index to activities_index
                    1  # get series from only column
                ]
            ).astype(uint64)

        return microdict_to_activity_series(self._collection_of_activity_dividend) / microdict_to_activity_series(
            self._collection_of_activity_divisor
        ).astype(  # Series division outrun mdict on numbers over 1000
            float32  # accuracy enough for metrics
        )

    def __warn_calculated_unsuccessfully_for_activity_diff(
        self, activity_durations_diff: ActivityDurationModifiedDifferential
    ) -> None:
        na_durations_activities = list(activity_durations_diff[activity_durations_diff.isna()].index)

        _ = [
            logger.warning(
                f"Продолжительность этапа равно нулю слишком часто или данных недостаточно, чтобы оценить этап {activity}."
            )
            for activity in na_durations_activities
        ]
