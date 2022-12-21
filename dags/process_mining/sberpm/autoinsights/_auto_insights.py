from collections import deque
from dataclasses import dataclass, field
from os.path import join as join_path
from re import findall, sub
from sys import stdout
from typing import Callable, Dict, Generator, Iterable, Mapping, Optional, Set, Tuple, Union

from loguru import logger
from methodtools import lru_cache

from numpy import argmax, array, float32, float64, int64, nan, nanmean, newaxis, ndarray, select, where
from numpy import any as np_any
from numpy import mean as np_mean
from numpy import median as np_median
from numpy import round as np_round
from numpy import sum as np_sum
from numpy import var as np_variance
from numpy.linalg import norm as linalg_norm
from pandas import Categorical, DataFrame, Series, Timestamp
from pandas.core.groupby.generic import DataFrameGroupBy

from nltk.stem import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest as IoF
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import OneClassSVM

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.autoinsights._auto_insights_metadesc import (
    NO_FIN_EFFECT_IMPLEMENTED_METRICS,
    MISTAKE_LEMMAS_PATH,
    REVERSAL_LEMMAS_PATH,
    DataHolderFeatures,
    ActivityInsightfulMetrics,
    ActivityInsightsOutcome,
    AnomalyDetectionParameter,
    FinEffectSummaryFormat,
    MessageToUser,
    MetricsFinEffectInterpretation,
    TextProblemTypes,
)
from process_mining.sberpm.autoinsights._data_sorting import DataHolderSorter
from process_mining.sberpm.autoinsights._influencing_activities import Influencing_activities
from process_mining.sberpm.autoinsights.time_metrics_calculation._duration_activity import ActivityDurationChangeInspector
from process_mining.sberpm.autoinsights.time_metrics_calculation._median_times_activity import ActivityDurationMeanMedianInspector

logger.remove()
logger.add(stdout, colorize=True, format="<level>{level}</level> | {name}: {message}")


def fillna_mean_efficiently(data: DataFrame) -> DataFrame:
    for column in data.columns[np_any(data.isnull())]:  # skip columns without nan
        data[column].fillna(np_mean(data[column]), inplace=True)  # just series mean

    return data


def transform_text_series_to_set_of_words(text_series: Series) -> Iterable[str]:
    return set(text_series.str.cat(sep=" ").split())


@dataclass
class AutoInsights:  # TODO add all properties
    """
    Something like MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)

    Parameters:
        data_holder(DataHolder): desc
        success_activity(Optional[str]): activity that represents successful process. Defaults to None.
        cluster_eps(float): common hyperparam for DBSCAN and IsolationForest.
            Defaults to 0.1.
        min_cost(float): second cost of human work. Defaults to 0.6.

    Args:
        _dbscan_eps(float): DBSCAN hyperparameter
        _contamination(float): IsolationForest hyperparameter

        __lemmatized_mistake_words(Iterable[str]): special lemmas from file representing common words
            for activities with errors
        __lemmatized_reversal_words(Iterable[str]):special lemmas from file representing common words
            for activities with retention
        __ignore_context_lemmatize_russian(Callable): russian word lemmatizer
        __ignore_context_lemmatize_latin(Callable): english word lemmatizer

        _res_with_clustering(DataFrame): table of activities on metrics
            (DataFrame(zip(activities, metrics)) -> True/False)
        _res(DatFrame): unprocessed table on raw metrics (DataFrame(zip(activities, raw_metrics)) -> True/False)
            (DataFrame(zip(activity_transition, metrics)) -> True/False)
        _transition_ai(DatFrame): table of activity_transitions on metrics
                    (DataFrame(zip(activity_transition, metrics)) -> True/False)
        __mistaken_processes(Iterable[int]): processes (id_columns) with errored activities
        __reversal_processes(Iterable[int]): processes (id_columns) with activities explaining retention
        __successful_processes(Iterable[int]): processes (id_columns)
            with successful terminal activity

    Examples:
    >>> from process_mining.sberpm.autoinsights import AutoInsights
    >>>
    >>> auto_i = AutoInsights(data_holder, success_activity='Stage_8', sec_cost=111)
    >>> auto_i.apply()
    >>>
    >>> auto_i.get_result()
    >>> auto_i.get_clustered_result()
    >>> auto_i.get_boolean_clustered_result()
    >>> auto_i.get_transition_ai()
    >>> print(auto_i.get_description())
    """

    data_holder: DataHolder
    success_activity: Optional[Union[int, str]] = None
    cluster_eps: float = 0.1
    min_cost: float = 0.6

    # * copy-pasting is bad :( will correct
    _timestamp_columns: Iterable[str] = field(init=False, repr=True)
    _id_name: str = field(init=False, repr=True)
    _activity_name: str = field(init=False, repr=True)
    _start_timestamp_name: Optional[str] = field(init=False, repr=True)
    _end_timestamp_name: Optional[str] = field(init=False, repr=True)
    _text_description_name: Optional[str] = field(init=False, repr=True)

    _dbscan_eps: float = field(init=False, repr=True)
    _contamination: float = field(init=False, repr=True)

    __lemmatized_mistake_words: Iterable[str] = field(init=False, repr=False)
    __lemmatized_reversal_words: Iterable[str] = field(init=False, repr=False)
    __ignore_context_lemmatize_russian: Callable = field(init=False, repr=False)
    __ignore_context_lemmatize_latin: Callable = field(init=False, repr=False)

    __unique_activities: Iterable[Union[int, str]] = field(init=False, repr=False)

    __successful_processes: Iterable[int] = field(init=False, repr=False)
    __mistaken_processes: Iterable[int] = field(init=False, repr=False)
    __reversal_processes: Iterable[int] = field(init=False, repr=False)

    _res: DataFrame = field(init=False, repr=True)
    _res_with_clustering: DataFrame = field(init=False, repr=False)
    _res_with_insight_outcomes: DataFrame = field(init=False, repr=False)
    _transition_ai: DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_name = self._get_id_name
        self._activity_name = self._get_activity_name
        self._start_timestamp_name = self._get_start_timestamp_name
        self._end_timestamp_name = self._get_end_timestamp_name
        self._text_description_name = self._get_text_description_name
        self._timestamp_columns = self._get_timestamp_columns

        self.__fill_nan_timestamp_with_max()
        self.__ensure_data_is_sorted_by_id_to_timestamp()
        self.__label_encode_id_column()
        self.__cast_categorical_autoinsights_data()

        self.sec_cost = self._get_sec_cost
        self.__unique_activities = self.data_holder.data[self._activity_name].values.categories

        self._dbscan_eps = self._calculate_dbscan_eps_or_contamination_by_cluster_eps_value(
            AnomalyDetectionParameter.dbscan_eps
        )
        self._contamination = self._calculate_dbscan_eps_or_contamination_by_cluster_eps_value(
            AnomalyDetectionParameter.contamination
        )
        self.__ensure_success_activity_is_present()

        self._res_with_clustering = DataFrame()  # Результат
        self._res_with_insight_outcomes = DataFrame()  # Результат
        self._res = DataFrame()  # Таблица с значениями отнормированных метрик на 1, по каждой ноде
        self._transition_ai = DataFrame()  # Аномальности переходов

        # Лемматизация
        self.__lemmatized_mistake_words = self._get_lemmatized_mistake_words
        self.__lemmatized_reversal_words = self._get_lemmatized_reversal_words
        self.__ignore_context_lemmatize_russian = self._get_ignore_context_lemmatize_russian
        self.__ignore_context_lemmatize_latin = self._get_ignore_context_lemmatize_latin

        self.__successful_processes = []
        self.__mistaken_processes = []
        self.__reversal_processes = []

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

    @property
    def _get_text_description_name(self) -> str:
        return self.data_holder.text_column

    # * TODO __get_timestamp_columns DataHolderSorter -> transfer DataHolder
    @property
    def _get_timestamp_columns(self) -> Iterable[str]:
        return self.data_holder.data.select_dtypes(
            ["datetime", "datetimetz"],
        ).columns  # start_timestamp, end_timestamp

    @property
    def __get_success_activity_as_most_common_terminal_activity(
        self,
    ) -> Union[int, str]:
        return (
            self.data_holder.data.groupby(self._id_name)[  # group by processes
                self._activity_name
            ]  # locate activities for every process
            .last()  # locate last (terminal) activity for every process
            .mode()  # get most common from terminal activities
        )[
            0
        ]  # get value of calculated series

    def __ensure_success_activity_is_present(self) -> None:
        if self.success_activity is None:
            self.success_activity = self.__get_success_activity_as_most_common_terminal_activity

    def __fill_nan_timestamp_with_max(self) -> None:
        for timestamp_name in self._timestamp_columns:
            timestamp_series = self.data_holder.data[timestamp_name]
            timezone = hasattr(timestamp_series.dtype, "tz") and timestamp_series.dtype.tz or None

            timestamp_series.fillna(Timestamp("2038-01-19 03:14:07", tz=timezone), inplace=True)

    def __ensure_data_is_sorted_by_id_to_timestamp(self) -> None:
        if not self.data_holder.data[self._id_name].is_monotonic_increasing:  # fast id check
            # to sort is tens times faster than to check datetime series are sorted, so give up checking
            DataHolderSorter(self.data_holder).sort_data_by_id_to_timestamp()

    def __label_encode_id_column(self):
        label_encoder = LabelEncoder()
        self.data_holder.data[self._id_name] = label_encoder.fit_transform(self.data_holder.data[self._id_name])

    def __cast_categorical_autoinsights_data(self) -> None:
        categorical_columns = (
            (self._id_name, self._activity_name, self._text_description_name)
            if self._text_description_name
            else (self._id_name, self._activity_name)
        )

        for categorical_feature_name in categorical_columns:
            self.data_holder.data[categorical_feature_name] = Categorical(
                self.data_holder.data[categorical_feature_name]
            )

    @property
    def _get_lemmatized_mistake_words(self) -> str:
        with open(join_path(MISTAKE_LEMMAS_PATH), encoding="utf-8") as file:
            return file.read().split("\n")

    @property
    def _get_lemmatized_reversal_words(self) -> str:
        with open(join_path(REVERSAL_LEMMAS_PATH), encoding="utf-8") as file:
            return file.read().split("\n")

    @property
    def _get_sec_cost(self) -> float:
        return self.min_cost / 60

    def __guarantee_valid_anomaly_detection_eps(self) -> None:
        if not 0 < self.cluster_eps < 1:
            self.cluster_eps = 0.001 if self.cluster_eps <= 0 else 0.999
            logger.warning(MessageToUser.correct_cluster_eps.format(self.cluster_eps))

    def _calculate_dbscan_eps_or_contamination_by_cluster_eps_value(
        self, attr_name: AnomalyDetectionParameter
    ) -> float:
        self.__guarantee_valid_anomaly_detection_eps()

        return (
            self.cluster_eps if attr_name == AnomalyDetectionParameter.dbscan_eps else 0.5 * (1 - self.cluster_eps)
        )

    @property
    def _get_dbscan_eps(self) -> float:
        return self._calculate_dbscan_eps_or_contamination_by_cluster_eps_value(
            AnomalyDetectionParameter.dbscan_eps
        )

    @property
    def _get_contamination(self) -> float:
        return self._calculate_dbscan_eps_or_contamination_by_cluster_eps_value(
            AnomalyDetectionParameter.contamination
        )

    # TODO refactor
    @property
    def __get_data_cleaned_of_successful_processes(self) -> DataHolder:
        successful_processes = self.data_holder.data[
            self.data_holder.data[self._activity_name]
            == self.success_activity  # filter out unsuccessful activities
        ][
            self._id_name
        ]  # slice processes where successful activity is

        data_holder_no_success = self.data_holder.copy()
        data_holder_no_success.data = self.data_holder.data[
            ~self.data_holder.data[self._id_name].isin(
                successful_processes,
            )  # processes with all of successful_processes
        ]  # negation of last

        return data_holder_no_success

    @property
    def _get_ignore_context_lemmatize_russian(self) -> Callable:
        pymorph_analyzer = MorphAnalyzer(lang="ru")

        return pymorph_analyzer.normal_forms

    @property
    def _get_ignore_context_lemmatize_latin(self) -> Callable:
        wordnet_lemmatizer = WordNetLemmatizer()
        wordnet_lemmatizer.lemmatize("tester")  # it is initialized by calling

        return wordnet_lemmatizer.lemmatize

    def __extract_lowered_words_of_text(self, text: str) -> str:
        # TODO name expression
        return sub(r"([^\W\d_]*)([\W_\d]+)([^\W\d_]*)", r"\1 \3", text).lower()

    def __retrieve_latin_doc_set(self, text: str) -> Iterable[str]:
        return set(findall(r"[A-z]+", self.__extract_lowered_words_of_text(text)))

    def __retrieve_russian_doc_set(self, text: str) -> Iterable[str]:
        return set(findall(r"[А-я]+", self.__extract_lowered_words_of_text(text)))

    @lru_cache(maxsize=2000)
    def __lemmatize_russian_word(self, word: str) -> str:
        return self.__ignore_context_lemmatize_russian(word)[0]

    @lru_cache(maxsize=1000)
    def __lemmatize_latin_word(self, word: str) -> str:
        verb_mode = "v"

        return self.__ignore_context_lemmatize_latin(word, verb_mode)

    def __retrieve_russian_lemmas_list(self, text: str) -> Iterable[str]:
        return [self.__lemmatize_russian_word(word) for word in self.__retrieve_russian_doc_set(text)]

    def __retrieve_latin_lemmas_list(self, text: str) -> Iterable[str]:
        lemmas_queue = deque()
        for word in self.__retrieve_latin_doc_set(text):
            lemmas_queue.append(self.__lemmatize_latin_word(word))

        return list(lemmas_queue)

    @lru_cache(maxsize=50)
    def __retrieve_set_of_lemmatized_doc(self, text: str) -> Iterable[str]:
        return {*self.__retrieve_russian_lemmas_list(text), *self.__retrieve_latin_lemmas_list(text)}

    @lru_cache(maxsize=5000)
    def __retrieve_set_of_lemmatized_word(self, word: str) -> Iterable[str]:
        return {self.__lemmatize_latin_word(word) if word.isascii() else self.__lemmatize_russian_word(word)}

    def __apply_text2problem_to_unique_texts(
        self, text_collection: Iterable[str], problem_type: TextProblemTypes
    ) -> Iterable[bool]:
        return (self.text2problem(text=text, problem_type=problem_type) for text in text_collection)

    def __apply_text2problem_to_unique_activities(self, problem_type: TextProblemTypes) -> Iterable[bool]:
        return (
            self.text2problem(text=activity, problem_type=problem_type) for activity in self.__unique_activities
        )

    def __take_mask_where_text_is(self, text: str) -> Series:
        return self.data_holder.data[self._text_description_name] == text

    def __guarantee_text_column_exists(self) -> None:
        if not self._text_description_name:
            return

    def __assign_mask_of_text_issue(self, problem_type: TextProblemTypes) -> ndarray:
        self.__guarantee_text_column_exists()

        if not self._text_description_name:
            return

        unique_texts = self.data_holder.data[self._text_description_name].values.categories

        self.data_holder.data[DataHolderFeatures[problem_type.name]] = select(
            [self.__take_mask_where_text_is(unique_text) for unique_text in unique_texts],
            [*self.__apply_text2problem_to_unique_texts(unique_texts, problem_type=problem_type)],
            default=self.data_holder.data[DataHolderFeatures[problem_type.name]],
        )

    def __assign_mask_of_activity_text_issue(self, problem_type: TextProblemTypes) -> ndarray:
        self.data_holder.data[DataHolderFeatures[problem_type.name]] = select(
            [*self.__apply_text2problem_to_unique_activities(problem_type=problem_type)],
            [True] * len(self.__unique_activities),
            default=False,
        )

    @lru_cache(maxsize=1)
    @property
    def __grouped_by_activity(self) -> DataFrameGroupBy:
        return self.data_holder.data.groupby(self._activity_name)

    @property
    def __is_text_column_activity_times(self) -> Iterable[bool]:
        return [bool(self._text_description_name)] * len(self.__unique_activities)

    @property
    def __activity_grouped_text_of_unique_words(self) -> Generator[Series, None, None]:
        return (
            self.__grouped_by_activity[self._text_description_name]
            .apply(transform_text_series_to_set_of_words)
            .str.join(sep=" ")
        )

    def __apply_text2problem_to_activity_grouped_text(
        self, problem_type: TextProblemTypes
    ) -> Generator[bool, None, None]:
        return (
            self.text2problem(text=activity_text, problem_type=problem_type)
            for activity_text in self.__activity_grouped_text_of_unique_words
        )

    def __calculate_textual_or_issue_mask_by_activity(
        self, textual_calculation: Iterable[bool], problem_type: TextProblemTypes
    ) -> Iterable[bool]:
        return (
            textual_calculation
            if self._text_description_name
            else self.__grouped_by_activity[DataHolderFeatures[problem_type.name]].all()
        )

    def __calculate_insight_metrics_by_issue_mask(self, problem_type: TextProblemTypes) -> ndarray:
        return self.__calculate_textual_or_issue_mask_by_activity(
            (
                [*self.__apply_text2problem_to_activity_grouped_text(problem_type)]
                if self._text_description_name
                else self.__is_text_column_activity_times
            ),
            problem_type=problem_type,
        )

    def __assign_text_metrics_for_clustering_result(
        self,
        issue_metrics: Iterable[ActivityInsightfulMetrics],
        problem_type: TextProblemTypes,
    ) -> None:
        self.__assign_mask_of_activity_text_issue(problem_type=problem_type)
        self._res_with_clustering = self._res_with_clustering.assign(
            **dict.fromkeys(issue_metrics, self.__calculate_insight_metrics_by_issue_mask(problem_type))
        )
        self.__assign_mask_of_text_issue(problem_type=problem_type)

    def _get_mean_time_features(
        self,
    ) -> Tuple[Dict[Union[int, str], float64], Dict[Union[int, str], int64], Dict[Union[int, str], float32]]:
        """
        Расчет
        1) отношения медианного времени и среднего времени пребывания в 1 вершине mean_median
        2) среднего времени пребывания в 1 вершине mean_time
        """
        duration_mean_median = ActivityDurationMeanMedianInspector(self.data_holder.copy())

        return (
            duration_mean_median.activity_duration_metrics,
            duration_mean_median.mean_times_microdict_to_activity_data_mean_times_dict,
            duration_mean_median.times_microdict_to_activity_data_times_dict,
        )

    def _get_no_successes_mean_time(self) -> Tuple[dict, dict]:
        """
        Расчет
        1) отношения медианного времени и среднего времени пребывания в 1 вершине mean_median
        2) среднего времени пребывания в 1 вершине mean_time
        """
        return ActivityDurationMeanMedianInspector(
            self.__get_data_cleaned_of_successful_processes
        ).mean_times_microdict_to_activity_data_mean_times_dict

    def _get_diff_time(self) -> Mapping[str, float32]:
        """
        Расчет отношения среднего времени пребывания в вершине во 2 половине рассматриваемого периода
        к 1 половине.
        """
        return ActivityDurationChangeInspector(self.data_holder.copy()).activity_duration_metrics

    def __compute_activities_frequency_metrics(
        self, grouped_data: DataFrame
    ) -> Tuple[Mapping[Union[int, str], float], Mapping[Union[int, str], float], Mapping[Union[int, str], int]]:
        """
        Расчет
        1) вероятности нахождения в вершине
        2) среднего количества вхождений вершины в процесс (аналогично 1), только с учетом дубликатов)
        """

        def get_sums_from_filter(filter_obj) -> Tuple[int]:
            filter_obj_list = list(filter_obj)

            return len(set(filter_obj_list)), len(filter_obj_list)

        frequency, duplicates_rate, counts = {}, {}, {}
        nodes = set(self.data_holder.data[self._activity_name])
        activity_grouped_data = grouped_data[self._activity_name]

        for node in nodes:

            def filter_current_activity_out(activities, current_activity=node):
                return get_sums_from_filter(filter(lambda activity: activity == current_activity, activities))

            count, count_full = map(
                sum,
                zip(*activity_grouped_data.apply(filter_current_activity_out).values),
            )

            frequency[node], duplicates_rate[node], counts[node] = (
                count / len(grouped_data),
                count_full / len(grouped_data),
                count,
            )

        return frequency, duplicates_rate, counts

    def __compute_repeatable_activity_count(self, grouped_data: DataFrame) -> Mapping[Union[int, str], float]:
        """
        Расчет количества повторяющихся активити
        """
        multiple = {}

        nodes = set(self.data_holder.data[self._activity_name])
        activity_grouped_data = grouped_data[self._activity_name]

        for node in nodes:

            def count_current_activity(activities, current_activity=node) -> Optional[int]:
                return len([*filter(lambda activity: activity == current_activity, activities)]) or nan

            multiple[node] = nanmean(activity_grouped_data.apply(count_current_activity))

        return multiple

    # TODO refactor
    def __compute_mean_activities_count(self, grouped_data: DataFrame) -> Mapping[Union[int, str], float]:
        """
        Расчет средней длины процесса, в котором участвует вершина
        """
        nodes = set(self.data_holder.data[self._activity_name])
        activity_grouped_data = grouped_data[self._activity_name]

        return {
            node: nanmean(
                activity_grouped_data.apply(lambda activities: len(activities) if node in activities else nan)
            )
            for node in nodes
        }

    def __retrieve_data_frame_from_metrics(
        self,
        **metrics: Mapping[str, Mapping[Union[int, str], Mapping[int, float]]],
    ) -> DataFrame:
        return DataFrame(
            metrics.values(),
            index=(
                "mean_time",
                "diff_time",
                "frequency",
                "cycles",
                "mean_median",
                "duplicates_rate",
                "activities_num",
                "anomaly_level",
            )[
                :-1
            ],  # TODO sth with anomaly level, preferably initialization
        ).T.dropna(how="all")

    def __adjust_duplicates_rate_metric(self, data: DataFrame) -> DataFrame:
        return data.assign(duplicates_rate=lambda record: record["duplicates_rate"] * record["mean_time"])

    def __adjust_frequency_metric(self, data: DataFrame) -> DataFrame:
        return data.assign(frequency=lambda record: 1 - record["frequency"])

    def __assign_with_anomaly_level(self, data: DataFrame) -> DataFrame:
        return data.assign(
            anomaly_level=lambda record: MinMaxScaler().fit_transform(
                linalg_norm(
                    array(record),
                    axis=1,
                )[..., newaxis]
            ),
        )

    def __construct_insightful_metrics_table(self, **metrics) -> None:
        _ = self.__retrieve_data_frame_from_metrics(**metrics)
        _.loc[:] = MinMaxScaler().fit_transform(_)

        self._res = self.__assign_with_anomaly_level(
            fillna_mean_efficiently(
                self.__adjust_frequency_metric(self.__adjust_duplicates_rate_metric(_)),
            )
        ).sort_values(by=["anomaly_level"], ascending=False)

    @property
    def __transition_tuples_series(self) -> ndarray:
        shifted_activity_series = self.data_holder.data[self._activity_name].shift(-1)
        shifted_id_series = self.data_holder.data[self._id_name].shift(-1)

        return Series(list(zip(self.data_holder.data[self._activity_name], shifted_activity_series))).where(
            self.data_holder.data[self._id_name] == shifted_id_series, nan
        )

    def __set_anomaly_level(self) -> None:
        self._res_with_insight_outcomes[ActivityInsightsOutcome.anomaly_level] = self._res["anomaly_level"].copy()

    def __set_fin_effect_sum(self) -> None:
        fin_effect_metrics = [
            metrics
            for metrics in ActivityInsightfulMetrics
            if metrics
            not in (
                *NO_FIN_EFFECT_IMPLEMENTED_METRICS,
                ActivityInsightfulMetrics.prolonged,  # FIXME
            )
        ]

        self._res_with_insight_outcomes[ActivityInsightsOutcome.fin_effect_amount] = np_sum(
            self._res_with_clustering[fin_effect_metrics], axis=1
        ).astype(int64)

    def __gather_successful_processes(self) -> None:
        self.__successful_processes = [
            *self.data_holder.data[
                self.data_holder.data[self.data_holder.activity_column]
                == self.success_activity  # filter out unsuccessful activities
            ][
                self.data_holder.id_column  # slice processes where successful activity is
            ].values
        ]

    def __retrieve_where_any_feature_in_process(
        self, groupby_id: DataFrameGroupBy, feature: str
    ) -> Iterable[Union[str, int, float]]:
        activity_hits = groupby_id[feature].any()

        return [
            *filter(
                None,
                where(
                    activity_hits.values,
                    activity_hits.index,
                    None,  # None gets along with both numbers and objects
                ),
            )
        ]

    def __gather_erroneous_processes(self) -> None:
        groupby_id = self.data_holder.data.groupby(self.data_holder.id_column)

        self.__mistaken_processes = self.__retrieve_where_any_feature_in_process(
            groupby_id, DataHolderFeatures.mistake
        )
        self.__reversal_processes = self.__retrieve_where_any_feature_in_process(
            groupby_id, DataHolderFeatures.reversal
        )

    def __gather_nice_or_nasty_processes(self) -> None:
        self.__gather_successful_processes()
        self.__gather_erroneous_processes()

    def _calc_transition_ai(self) -> DataFrame:
        act1 = self.data_holder.data[self.data_holder.activity_column]
        act2 = self.data_holder.data[self.data_holder.activity_column].shift(-1)
        id1 = self.data_holder.data[self.data_holder.id_column]
        id2 = self.data_holder.data[self.data_holder.id_column].shift(-1)
        tr_data = self.data_holder.data
        group_column = "transition"
        tr_data[group_column] = list(zip(act1, act2))
        tr_data = tr_data[id1 == id2]
        tr_data = tr_data[group_column].unique()
        temp_list = [
            (self._res.loc[trans[0]]["anomaly_level"] + self._res.loc[trans[1]]["anomaly_level"]) / 2.0
            for trans in tr_data
        ]

        self._transition_ai = DataFrame(index=tr_data, data={ActivityInsightsOutcome.anomaly_level: temp_list})

    # @lru_cache(maxsize=1)
    def apply(self) -> None:
        mean_median, mean_time, activity_times = self._get_mean_time_features()
        diff_time = self._get_diff_time()

        if self._start_timestamp_name is not None and self._end_timestamp_name is not None:
            grouped_data = self.data_holder.get_grouped_data(
                self._activity_name,
                self._start_timestamp_name,
                self._end_timestamp_name,
            )
        elif self._start_timestamp_name is not None:
            grouped_data = self.data_holder.get_grouped_data(self._activity_name, self._start_timestamp_name)
        else:
            grouped_data = self.data_holder.get_grouped_data(self._activity_name, self._end_timestamp_name)

        frequency, duplicates_rate, counts = self.__compute_activities_frequency_metrics(grouped_data=grouped_data)

        cycles = self.__compute_repeatable_activity_count(grouped_data=grouped_data)
        activities_num = self.__compute_mean_activities_count(grouped_data=grouped_data)

        grouped_data = grouped_data.iloc[:0]  # clear dataframe

        # TODO make separate class with following parameters
        self.__construct_insightful_metrics_table(
            mean_time=mean_time,
            diff_time=diff_time,
            frequency=frequency,
            cycles=cycles,
            mean_median=mean_median,
            duplicates_rate=duplicates_rate,
            activities_num=activities_num,
        )

        list_metric = [
            ActivityInsightfulMetrics.prolonged,
            ActivityInsightfulMetrics.monotonically_increasing_duration,
            ActivityInsightfulMetrics.irregular,
            ActivityInsightfulMetrics.has_loops,
            ActivityInsightfulMetrics.tight_bottleneck,
            ActivityInsightfulMetrics.distinctive_incidents_affect_duration,
            ActivityInsightfulMetrics.duration_increase_of_process_and_or_related_activities,
        ]

        metric_anomally = DataFrame(columns=list_metric, index=self._res.index, dtype=float64)

        best_params_db_scan = dict(
            algorithm="auto",
            eps=self._dbscan_eps,
            metric="euclidean",
            min_samples=1,
            n_jobs=-1,
        )

        right = DBSCAN().set_params(**best_params_db_scan)
        labels_dbscan = right.fit_predict(array(self._res["frequency"]).reshape(-1, 1))
        class_biggest_outlier = labels_dbscan[argmax(self._res["frequency"].values)]

        for node, label in zip(self._res.index, labels_dbscan):
            metric_anomally.at[node, ActivityInsightfulMetrics.irregular] = float(
                label == class_biggest_outlier
            )  # False -> 0
        for metric in self._res.drop(columns=["anomaly_level", "frequency"], axis=1).columns:
            right = IoF(contamination=self._contamination, n_jobs=-1, random_state=42, warm_start=True)
            labels_iof = right.fit_predict(array(self._res[metric]).reshape(-1, 1))
            col_df_rating_median = np_median(self._res[metric])

            for node, value, label in zip(self._res.index, self._res[metric], labels_iof):
                metric_anomally.at[node, list_metric[self._res.columns.get_loc(metric)]] = float(
                    label == -1 and value > col_df_rating_median
                )

        self._res_with_clustering = metric_anomally

        # Меняем алгоритм определения зацикленности, если cycles > 0, то зациклено
        self._res_with_clustering[ActivityInsightfulMetrics.has_loops] = (self._res["cycles"].copy() > 0).astype(
            float64
        )

        # Меняем алгоритм определения Bottle neck с низкой вариативностью ActivityInsightfulMetrics.distinctive_incidents_affect_duration
        self._res_with_clustering[ActivityInsightfulMetrics.widespread_bottleneck] = 0.0
        self._res_with_clustering[ActivityInsightfulMetrics.numerous_incident_stagnation] = 0.0

        for activity, value in mean_median.items():
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.prolonged):
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.tight_bottleneck] = float(
                    0.9 <= value <= 1.1
                )
                self._res_with_clustering.at[
                    activity, ActivityInsightfulMetrics.distinctive_incidents_affect_duration
                ] = float(not (0.9 <= value <= 1.1))
            else:
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.tight_bottleneck] = 0.0
                self._res_with_clustering.at[
                    activity, ActivityInsightfulMetrics.distinctive_incidents_affect_duration
                ] = 0.0
        # Меняем алгоритм определения Рост времени процесса и прочих этапов вследствие влияния данного этапа
        # анализирую результат SequentialFeatureSelector по времени и появлению этапов
        sfs = Influencing_activities(self.data_holder, mode="fast", metric="time", metric_f="appearance")
        influencing_activities = sfs.activities_impact()

        for activity in self._res_with_clustering.index:
            self._res_with_clustering.at[
                activity, ActivityInsightfulMetrics.duration_increase_of_process_and_or_related_activities
            ] = float(influencing_activities[activity] > np_round(self.__unique_activities.size * 0.2))

        # Считаем финэффект от Не регулярный этап как
        # (count - cycles) * mean_time * sec_cost
        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.irregular):
                financial_effect = (
                    (counts[activity] - self._res._get_value(activity, "cycles"))
                    * mean_time[activity]
                    * self.sec_cost
                )
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.irregular] = float(
                    financial_effect
                )
        # Считаем финэффект от Зацикленность как
        # cycles * mean_time * sec_cost
        for activity in self._res_with_clustering.index:
            financial_effect = self._res._get_value(activity, "cycles") * mean_time[activity] * self.sec_cost
            self._res_with_clustering.at[activity, ActivityInsightfulMetrics.has_loops] = float(financial_effect)
        # Финэффект от Разовых инцидентов
        # Выбираем выбросное время по одному этапу процесса, берем среднее время
        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(
                activity, ActivityInsightfulMetrics.distinctive_incidents_affect_duration
            ) and np_any(activity_times.get(activity)):
                X = array(activity_times[activity]).reshape(-1, 1)
                clf = OneClassSVM().fit(X)
                mask = array(clf.predict(X) - 1, dtype=bool)

                financial_effect = np_sum(X[mask]) * self.sec_cost
                self._res_with_clustering.at[
                    activity, ActivityInsightfulMetrics.distinctive_incidents_affect_duration
                ] = float(financial_effect)

        # Финэффект от Bottle neck с низкой вариативностью
        time_no_successes = self._get_no_successes_mean_time()

        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.tight_bottleneck):
                financial_effect = abs(
                    sum(
                        activity_times.get(activity, array([0]))
                        - time_no_successes.get(
                            activity,
                            activity_times.get(activity, array([0])),
                        ),
                    )
                    * self.sec_cost
                )

                X = array(activity_times.get(activity, array([0]))).reshape(-1, 1)
                X = MinMaxScaler().fit_transform(X)
                x_median = np_median(X)
                x_mean = np_mean(X)
                var = np_variance(X)

                try:
                    if x_mean / x_median > 1.15:
                        self._res_with_clustering.at[
                            activity, ActivityInsightfulMetrics.numerous_incident_stagnation
                        ] = float(financial_effect)
                        self._res_with_clustering.at[activity, ActivityInsightfulMetrics.tight_bottleneck] = 0.0
                    elif var < 0.125:
                        self._res_with_clustering.at[activity, ActivityInsightfulMetrics.tight_bottleneck] = float(
                            financial_effect
                        )
                    else:
                        self._res_with_clustering.at[
                            activity, ActivityInsightfulMetrics.widespread_bottleneck
                        ] = float(financial_effect)
                        self._res_with_clustering.at[activity, ActivityInsightfulMetrics.tight_bottleneck] = 0.0
                except Exception as err:
                    self._res_with_clustering.at[activity, ActivityInsightfulMetrics.tight_bottleneck] = float(
                        financial_effect
                    )
                    self._res_with_clustering.at[activity, ActivityInsightfulMetrics.widespread_bottleneck] = 0.0
                    self._res_with_clustering.at[
                        activity, ActivityInsightfulMetrics.numerous_incident_stagnation
                    ] = 0.0
                    logger.error(MessageToUser.null_duration_median.format(err=err, activity=activity))

        # Финэффект от Высокая длительность этапа
        for activity in self._res_with_clustering.index:
            financial_effect = self._res_with_clustering._get_value(
                activity, ActivityInsightfulMetrics.distinctive_incidents_affect_duration
            ) + self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.tight_bottleneck)
            self._res_with_clustering.at[activity, ActivityInsightfulMetrics.prolonged] = float(financial_effect)

        # Столбцы А и Б по ошибкам в текстовом поле и stage
        self._res_with_clustering[ActivityInsightfulMetrics.system_errors_stagnation] = 0.0
        self._res_with_clustering[ActivityInsightfulMetrics.system_errors_fail] = 0.0
        self._res_with_clustering[ActivityInsightfulMetrics.system_design_errors_fail] = 0.0

        self._res_with_clustering[ActivityInsightfulMetrics.reversal_stagnation] = 0.0
        self._res_with_clustering[ActivityInsightfulMetrics.reversal_fail] = 0.0
        # extra column in dataframe is represented problem in log line(in stage)

        # Mistake problems check
        self.__assign_text_metrics_for_clustering_result(
            issue_metrics=[
                ActivityInsightfulMetrics.reversal_stagnation,
                ActivityInsightfulMetrics.reversal_fail,
            ],
            problem_type=TextProblemTypes.reversal,
        )

        # Reversal problems check
        self.__assign_text_metrics_for_clustering_result(
            issue_metrics=[
                ActivityInsightfulMetrics.system_errors_stagnation,
                ActivityInsightfulMetrics.system_errors_fail,
            ],
            problem_type=TextProblemTypes.mistake,
        )

        self.data_holder.check_or_calc_duration()

        self.__gather_nice_or_nasty_processes()

        # A1 fin effect
        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.system_errors_stagnation):
                total_duration = np_sum(
                    self.data_holder.data[DataHolderFeatures.duration][
                        (self.data_holder.data[self._id_name].isin(self.__successful_processes))
                        & (self.data_holder.data[self._activity_name] == activity)
                        & (self.data_holder.data[DataHolderFeatures.mistake])
                    ]
                )
                financial_effect = total_duration * self.sec_cost
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.system_errors_stagnation] = float(
                    financial_effect
                )

        # A2 fin effect
        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.system_errors_fail):
                total_duration = np_sum(
                    self.data_holder.data[DataHolderFeatures.duration][
                        (~self.data_holder.data[self._id_name].isin(self.__successful_processes))
                        & (self.data_holder.data[self._id_name].isin(self.__mistaken_processes))
                        & (self.data_holder.data[self._activity_name] == activity)
                    ]
                )
                financial_effect = total_duration * self.sec_cost
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.system_errors_fail] = float(
                    financial_effect
                )

        # Б fin effect
        for activity in self._res_with_clustering.index:
            total_duration = np_sum(
                self.data_holder.data[DataHolderFeatures.duration][
                    (~self.data_holder.data[self._id_name].isin(self.__successful_processes))
                    & (~self.data_holder.data[self._id_name].isin(self.__mistaken_processes))
                    & (self.data_holder.data[self._activity_name] == activity)
                ]
            )
            financial_effect = total_duration * self.sec_cost
            self._res_with_clustering.at[activity, ActivityInsightfulMetrics.system_design_errors_fail] = float(
                financial_effect
            )

        # A1 fin effect Замедление процесса вследствие сторнирования на данном этапе
        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.reversal_stagnation):
                total_duration = np_sum(
                    self.data_holder.data[DataHolderFeatures.duration][
                        (self.data_holder.data[self._id_name].isin(self.__successful_processes))
                        & (self.data_holder.data[self._activity_name] == activity)
                        & (self.data_holder.data[DataHolderFeatures.reversal] is True)
                    ]
                )
                financial_effect = total_duration * self.sec_cost
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.reversal_stagnation] = float(
                    financial_effect
                )

        # A2 fin effect Неуспех процесса вследствие сторнирования на данном этапе
        for activity in self._res_with_clustering.index:
            if self._res_with_clustering._get_value(activity, ActivityInsightfulMetrics.reversal_fail):
                total_duration = np_sum(
                    self.data_holder.data[DataHolderFeatures.duration][
                        (~self.data_holder.data[self._id_name].isin(self.__successful_processes))
                        & (self.data_holder.data[self._id_name].isin(self.__reversal_processes))
                        & (self.data_holder.data[self._activity_name] == activity)
                    ]
                )

                financial_effect = total_duration * self.sec_cost
                self._res_with_clustering.at[activity, ActivityInsightfulMetrics.reversal_fail] = float(
                    financial_effect
                )

        self._res_with_clustering = self._res_with_clustering.transform(abs)  # FIXME FIXME mean_time overflow
        self.__set_anomaly_level()
        self.__set_fin_effect_sum()

        self._calc_transition_ai()

        logger.info(MessageToUser.algorithm_succeeded)

    def __ensure_problem_type(self, problem_type: str) -> None:
        if not isinstance(problem_type, TextProblemTypes):
            raise RuntimeError("""problem_type должен быть из TextProblemTypes""")

    def __extract_lemmatized_set(self, text: str) -> Set[str]:
        not_a_single_word = " " in text

        if not_a_single_word:
            return self.__retrieve_set_of_lemmatized_doc(text)

        return self.__retrieve_set_of_lemmatized_word(text)

    def __choose_lemmatized_words_by_problem(self, problem: str) -> Iterable[str]:
        return (
            self.__lemmatized_mistake_words
            if problem == TextProblemTypes.mistake
            else self.__lemmatized_reversal_words
        )

    def text2problem(self, text, problem_type=TextProblemTypes.mistake) -> bool:
        """
        Compare lemmatized problem words and text.
        Return True if text is problematic
        """
        self.__ensure_problem_type(problem_type)

        return bool(
            self.__extract_lemmatized_set(text) & set(self.__choose_lemmatized_words_by_problem(problem_type))
        )

    def __guarantee_apply_method_called(self, condition_for_failure: bool) -> None:
        if condition_for_failure:
            logger.warning(MessageToUser.algorithm_should_run)
            self.apply()

    def get_result(self) -> DataFrame:
        self.__guarantee_apply_method_called(condition_for_failure=self._res.empty)

        return self._res

    def get_clustered_result(self) -> DataFrame:
        self.__guarantee_apply_method_called(condition_for_failure=self._res_with_clustering.empty)

        return (
            self._res_with_clustering.astype(int64)
            .assign(
                **dict.fromkeys(
                    NO_FIN_EFFECT_IMPLEMENTED_METRICS,
                    0,
                )
            )
            .join(self._res_with_insight_outcomes, how="inner")
        )

    def get_boolean_clustered_result(self) -> DataFrame:
        self.__guarantee_apply_method_called(condition_for_failure=self._res_with_clustering.empty)

        return self._res_with_clustering.astype(bool).join(self._res_with_insight_outcomes, how="inner")

    def __make_out_insight_financial_interpretation(
        self, metrics_series: Series
    ) -> MetricsFinEffectInterpretation:
        insightful_metrics = metrics_series.name

        return MetricsFinEffectInterpretation[insightful_metrics.name]

    def __format_metrics_fin_effect(
        self,
        effect_interpretation: MetricsFinEffectInterpretation,
        fin_amount: float64,
        fin_activities: str,
    ) -> str:
        return (
            f"{effect_interpretation}"
            f"{FinEffectSummaryFormat.fin_amount_indent}{fin_amount} {FinEffectSummaryFormat.money_unit}:"
            f"{FinEffectSummaryFormat.fin_activity_indent}{fin_activities}"
        )

    def __wrap_financial_insights_summary(self, fin_summary: DataFrame, interpretation_sep: str):
        return fin_summary.apply(
            lambda insightful_metrics: self.__format_metrics_fin_effect(
                effect_interpretation=self.__make_out_insight_financial_interpretation(insightful_metrics),
                fin_amount=insightful_metrics.fin_amount,
                fin_activities=insightful_metrics.fin_activities,
            ),
            axis=1,
        ).str.cat(sep=interpretation_sep)

    def __construct_finance_metrics_summary(self, activities_by_insightful_metrics: DataFrame) -> Iterable[str]:
        def mark_valuable_activities_of_metric(metrics_values_at_activity):
            return metrics_values_at_activity[metrics_values_at_activity != 0].index

        # TODO namedtuple
        metrics_fin_effect_amount = np_sum(activities_by_insightful_metrics, axis=1)
        fin_metrics_to_activities = activities_by_insightful_metrics.apply(
            mark_valuable_activities_of_metric,
            axis=1,
        ).str.join(FinEffectSummaryFormat.fin_activity_sep)

        return self.__wrap_financial_insights_summary(
            DataFrame(
                dict(
                    fin_activities=fin_metrics_to_activities,
                    fin_amount=metrics_fin_effect_amount,
                )
            )[metrics_fin_effect_amount != 0],
            interpretation_sep=FinEffectSummaryFormat.interpretation_sep,
        )

    def __construct_finance_complex_summary(self, activities_by_insightful_metrics: DataFrame) -> str:
        # TODO namedtuple
        complex_fin_amount = np_sum(self._res_with_insight_outcomes[ActivityInsightsOutcome.fin_effect_amount])
        activities_with_finance = activities_by_insightful_metrics.columns[
            activities_by_insightful_metrics.any()
        ].str.cat(sep=", ")

        complex_summary = self.__format_metrics_fin_effect(
            effect_interpretation=MetricsFinEffectInterpretation.complex_finance,
            fin_amount=complex_fin_amount,
            fin_activities=activities_with_finance,
        )

        return f"{FinEffectSummaryFormat.interpretation_sep}{FinEffectSummaryFormat.complex_effect_sep}{complex_summary}"

    def get_description(self) -> str:
        activities_by_insightful_metrics = self.get_clustered_result()[[*ActivityInsightfulMetrics]].T

        finance_metrics_summary = self.__construct_finance_metrics_summary(
            activities_by_insightful_metrics,
        )
        finance_complex_summary = self.__construct_finance_complex_summary(
            activities_by_insightful_metrics,
        )

        return f"{finance_metrics_summary}{finance_complex_summary}"

    @property
    def __activity_transitions_name(self) -> str:
        return f"Переходы {self._activity_name}"

    def __cast_transition_tuples_to_string(self, activity_transitions_series: Series) -> Series:
        activity_separator = " ⏩ "

        return activity_transitions_series.dropna().apply(activity_separator.join)  # dropna is fastest

    def __create_activity_transitions_features(self, activity_transitions_series: Series) -> Mapping[str, Series]:
        return {
            self._start_timestamp_name: self.data_holder.data[self._end_timestamp_name],
            self._end_timestamp_name: self.data_holder.data[self._start_timestamp_name].shift(-1, fill_value=nan),
            self.__activity_transitions_name: self.__cast_transition_tuples_to_string(activity_transitions_series),
        }

    @property
    def __data_holder_with_transitions_activity(self) -> DataHolder:
        activity_transitions = self.__transition_tuples_series

        return self.data_holder.data.assign(
            **self.__create_activity_transitions_features(activity_transitions)
        ).dropna(
            subset=self.__activity_transitions_name,
        )  # transitions values are nan where processes are shifting, at last activity

    def __inform_user_transitions_wont_be_analyzed(self) -> None:
        logger.error(MessageToUser.transitions_require_both_timestamps)

    def __cannot_analyze_transitions(self) -> bool:
        if not self._start_timestamp_name or not self._end_timestamp_name:
            self.__inform_user_transitions_wont_be_analyzed()
            return True

        return False

    def get_transition_ai(self) -> DataFrame:
        if self.__cannot_analyze_transitions():
            return

        if not self._transition_ai.empty:
            return self._transition_ai.copy()
        logger.warning("Для вывода результата необходимо сначала вызвать метод apply. Метод будет вызван неявно.")
        self.apply()
