from dataclasses import dataclass
from typing import Generic, TypeVar, Union
from warnings import warn

from numpy import sort as np_sort
from pandas import Index

from process_mining.sberpm._holder import DataHolder


class TimestampColumnNames(
    Generic[
        TypeVar(
            "Index",
            bound=Union[("DataHolder_inst.start_timestamp_column", "DataHolder_inst.end_timestamp_column")],
        )
    ]
):
    pass


@dataclass
class DataHolderSorter:
    data_holder: DataHolder

    def sort_data_by_id_to_timestamp(self) -> None:
        id_name = self.data_holder.id_column
        timestamp_columns = self._get_timestamp_columns

        self.__sort_data_by_id_and_timestamp(id_name, timestamp_columns)

    # TODO transfer to DataHolder
    @property
    def _get_timestamp_columns(self) -> TimestampColumnNames:
        return self.data_holder.data.select_dtypes(
            ["datetime", "datetimetz"],
        ).columns

    def __sort_series_with_numpy(self, column_name: str, ascending=True) -> None:
        """slow for timestamp, though a bit faster in general"""
        self.data_holder.data[column_name].values = np_sort(
            self.data_holder.data[column_name].values, kind="quicksort"
        )
        if not ascending:
            self.data_holder.data[column_name].values = self.data_holder.data[column_name].values[::-1]

    def __sort_data_by_id_solely(self) -> None:
        self.__sort_series_with_numpy(column_name=self.data_holder.id_column)
        warn("DataHolder: time column is not given, cannot sort the activities.", UserWarning)

    def __sort_data_by_timestamp_and_optionally_by_id(self, id_name: str, timestamp_names: str) -> None:
        columns_to_sort_by = Index((id_name, *timestamp_names)) if id_name else Index((*timestamp_names,))

        self.data_holder.data.sort_values([*columns_to_sort_by], inplace=True)

    def __sort_data_by_id_and_timestamp(self, id_name: str, timestamp_names: str) -> None:
        if id_name and not timestamp_names.any():  # has process IDs only
            self.__sort_data_by_id_solely()
        elif timestamp_names.any():  # at least one timestamp column
            self.__sort_data_by_timestamp_and_optionally_by_id(id_name, timestamp_names)
        else:  # no timestamp columns
            raise RuntimeError(
                "The log that does not have an id and any of start timestamp or end timestamp columns cannot be processed."
            )
