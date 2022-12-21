from typing import Tuple

import pandas as pd

import plotly.express as px

from process_mining.sberpm.ml.graph_structure_prediction import generate_lags, select_features, search_model, predict
from process_mining.sberpm._holder import DataHolder


class GSPredictor:
    """
    Algorithm that predicts the graph structure,
    more specifically, if calculates the probabilities and average durations
    of nodes and edges for particular time periods and treats them
    as time series. Therefore, the algorithm can predict these values
    in the future.

    The steps of the algorithm:

    I. Data preparation

    1. Calculate time series for nodes and edges
        a) probability of nodes/edges
           (sum of probabilities of all nodes/edges during a time period == 1;
            or == 0 if no nodes/edges were executed during the time period)
        b) average time duration of node/edge in seconds

    2. Transform time series into lags matrices and add additional information:
        - differences of lags columns
        - one-hot encoding of months (only if one point of time series represents
          a day or a month)
        - number of working days in a month (only if one point of time series
          represents a month)

    II. Prediction (following steps are done for each time series)

    0. Data is split into 3 parts: "train", "test", "val".
    1. Gridsearch is used for several ml models to find the best one for
       train/test data.
    2. Several feature selection models are used to select best features
       for the model found in the previous step and for train/test data.
    3. Basically, the step 1 is done again (choosing the best model)
       but with some differences:
        a) more models are used: same ml models + Sarimax and Holt
        b) different data is used for training/testing: train+test/val
        c) the data has "the best" features only, found in the previous step
    4. Features from step 2 (and 3) and the model from step 3 are used
       to predict the result.

    Attributes
    ------
    DataFrames representing time series for nodes and edges
    (given data and predictions)

    nodes_prob: pd.DataFrame
    nodes_duration: pd.DataFrame
    edges_prob: pd.DataFrame
    edges_duration: pd.DataFrame

    nodes_prob_pred: pd.DataFrame
    nodes_duration_pred: pd.DataFrame
    edges_prob_pred: pd.DataFrame
    edges_duration_pred: pd.DataFrame

    Parameters
    ----------
    data_holder: DataHolder

    test_size: float, default=0.1
        Test size, part of the total data.

    val_size: float, default=0.1
        Val size, part of the total data.

    pred_period: int, default=5
        Number of points of time series to predict.

    period_type: {'D', 'M', 'Y'}
        One point of a time series: day month, year.

    two_models: bool, default=False
        If True, try all combinations of 2 ml models for predictions
        as well and unite their result. If False, models are used separately.

    edges: bool, default=False
        If True, predict time series for edges. If False, predictions
        is done for nodes only.

    refit: bool, default=False
        If True, refit the model at the each step of final prediction
        using the given data and the already predicted results.

    quick_mod: bool, default=False
        If True, the best model will be searched among larger set of regressor ML models.
    """

    def __init__(
        self,
        data_holder: DataHolder,
        test_size: float = 0.1,
        val_size: float = 0.1,
        pred_period: int = 5,
        period_type="D",
        two_models: bool = False,
        edges: bool = True,
        refit: bool = False,
        quick_mod: bool = False,
    ):
        self._dh = data_holder
        self._test_size = test_size
        self._val_size = val_size
        self._pred_period = pred_period
        self._period_type = period_type
        self._two_models = two_models

        self._edges = edges
        self._refit = refit
        self._quick_mod = quick_mod

    def apply(self) -> None:
        dh = self._dh
        timestamp_column = dh.get_timestamp_col()
        dh.check_or_calc_duration()

        df = create_rounded_date_column(dh.data, timestamp_column, self._period_type)
        date_range = pd.date_range(
            df["round_date"].min(),
            df["round_date"].max(),
            freq=self._period_type if self._period_type == "D" else f"{self._period_type}S",
        )
        # D, MS, YS (S - start)

        self.nodes_prob, self.nodes_duration = create_nodes_ts(df, dh.id_column, dh.activity_column, date_range)
        if self._edges:
            self.edges_prob, self.edges_duration = create_edges_ts(
                df, dh.id_column, dh.activity_column, date_range
            )
        else:
            self.edges_prob, self.edges_duration = None, None

        # --------------- PREDICTION --------------------
        self.nodes_prob_pred = predict_ts(
            self.nodes_prob,
            self._period_type,
            self._pred_period,
            self._test_size,
            self._val_size,
            self._two_models,
            norm=True,
            refit=self._refit,
            quick_mod=self._quick_mod,
        )
        self.nodes_duration_pred = predict_ts(
            self.nodes_duration,
            self._period_type,
            self._pred_period,
            self._test_size,
            self._val_size,
            self._two_models,
            norm=False,
            refit=self._refit,
            quick_mod=self._quick_mod,
        )
        if self._edges:
            self.edges_prob_pred = predict_ts(
                self.edges_prob,
                self._period_type,
                self._pred_period,
                self._test_size,
                self._val_size,
                self._two_models,
                norm=True,
                refit=self._refit,
                quick_mod=self._quick_mod,
            )
            self.edges_duration_pred = predict_ts(
                self.edges_duration,
                self._period_type,
                self._pred_period,
                self._test_size,
                self._val_size,
                self._two_models,
                norm=False,
                refit=self._refit,
                quick_mod=self._quick_mod,
            )
        else:
            self.edges_prob_pred, self.edges_duration_pred = None, None

    def plot_nodes_prob(self):
        """
        Plots time series for nodes' probabilities (given data and prediction).
        """
        plot(self.nodes_prob, self.nodes_prob_pred, "nodes_prob", "Nodes")

    def plot_nodes_duration(self):
        """
        Plots time series for nodes' durations (given data and prediction).
        """
        plot(self.nodes_duration, self.nodes_duration_pred, "nodes_duration", "Nodes")

    def plot_edges_prob(self):
        """
        Plots time series for edges' probabilities (given data and prediction).
        """
        if self.edges_prob_pred is not None:
            plot(self.edges_prob, self.edges_prob_pred, "edges_prob", "Edges")
        else:
            raise RuntimeError("Edges' time series were not calculated.")

    def plot_edges_duration(self):
        """
        Plots time series for edges' durations (given data and prediction).
        """
        if self.edges_duration_pred is not None:
            plot(self.edges_duration, self.edges_duration_pred, "edges_duration", "Edges")
        else:
            raise RuntimeError("Edges' time series were not calculated.")


def plot(ts: pd.DataFrame, ts_pred: pd.DataFrame, title: str, legend_title: str):
    df_all = pd.concat([ts, ts_pred], axis=0)
    if type(df_all.columns[0]) == tuple:
        df_all.columns = [f"{c[0]}->{c[1]}" for c in df_all.columns]  # for edges to avoid error error
    fig = px.line(df_all, y=df_all.columns, markers=True)
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value", legend_title=legend_title)
    t11 = ts.index.max()
    t12 = ts_pred.index.min()
    t1 = t11 + (t12 - t11) / 2  # between last known point and first pred point
    t21 = ts_pred.index[-2]
    t22 = ts_pred.index.max()
    t2 = t21 + (t22 - t21) * 1.5  # between last pred point and the one after (that does not exist)

    fig.add_vrect(x0=t1, x1=t2, annotation_text="prediction", fillcolor="green", opacity=0.15, line_width=0)
    fig.show()


def create_rounded_date_column(df: pd.DataFrame, timestamp_column: str, period_type: str) -> pd.DataFrame:
    if period_type in ["D", "M", "Y"]:
        if period_type == "D":
            round_date = df[timestamp_column].dt.round("D")
        else:
            round_date = df[timestamp_column].dt.to_period(period_type).dt.to_timestamp()  # round() doesn't work
    else:
        raise ValueError(f'Period type must be "D", "M", or "Y", but got "{period_type}"')

    # Delete first or/and last months if they contain the number of days less than their halves
    if period_type == "M":
        first_month_mask = round_date == round_date.min()
        if df[first_month_mask][timestamp_column].dt.day.min() > round_date.min().daysinmonth // 2:
            df = df[~first_month_mask]
            round_date = round_date[~first_month_mask]

        last_month_mask = round_date == round_date.max()
        if df[last_month_mask][timestamp_column].dt.day.max() < round_date.min().daysinmonth // 2:
            df = df[~last_month_mask]
            round_date = round_date[~last_month_mask]
    df["round_date"] = round_date
    return df


def create_nodes_ts(
    df: pd.DataFrame, idx: str, nodes_column: str, date_range: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_count = pd.DataFrame(index=date_range)
    nodes_duration = pd.DataFrame(index=date_range)

    for node in df[nodes_column].unique():
        dfg = (
            df[df[nodes_column] == node]
            .groupby("round_date")
            .agg({idx: "count", "duration": "mean"})
            .rename(columns={idx: "count"})
        )
        nodes_count = nodes_count.join(dfg["count"].rename(node))
        nodes_duration = nodes_duration.join(dfg["duration"].rename(node))

    nodes_count = normalize_rows(nodes_count.fillna(0)).fillna(0)  # 1st fillna - after joining, 2nd - after norm
    nodes_duration = nodes_duration.fillna(0)
    return nodes_count, nodes_duration


def create_edges_ts(
    df: pd.DataFrame, idx: str, nodes_column: str, date_range: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    edges_count = pd.DataFrame(index=date_range)
    edges_duration = pd.DataFrame(index=date_range)

    df["act2"] = df[nodes_column].shift(-1)
    df = df[df[idx] == df[idx].shift(-1)]
    df["edge"] = tuple(zip(df[nodes_column], df["act2"]))
    df = df.drop([nodes_column, "act2"], axis=1)

    for edge in df["edge"].unique():
        dfg = (
            df[df["edge"] == edge]
            .groupby("round_date")
            .agg({idx: "count", "duration": "mean"})
            .rename(columns={idx: "count"})
        )
        edges_count = edges_count.join(dfg["count"].rename(f"{edge[0]}-{edge[1]}"))
        edges_duration = edges_duration.join(dfg["duration"].rename(f"{edge[0]}-{edge[1]}"))

    edges_count = normalize_rows(edges_count.fillna(0)).fillna(0)
    edges_duration = edges_duration.fillna(0)
    return edges_count, edges_duration


def predict_ts(
    ts_dataframe,
    period_type,
    pred_period,
    test_size,
    val_size,
    two_models,
    norm: bool,
    refit: bool,
    quick_mod: bool,
) -> pd.DataFrame:
    predictions = []
    for obj_name in ts_dataframe.columns:
        ts = ts_dataframe[obj_name]
        X, y = generate_lags(ts=ts, period_type=period_type)

        best_model, _ = search_model(
            X,
            y,
            test_size,
            val_size,
            period_type=period_type,
            non_ml_models=False,
            two_models=two_models,
            quick_mod=quick_mod,
        )
        best_indexes = select_features(X, y, test_size, val_size, best_model)
        X_best = X.iloc[:, best_indexes]
        best_model, best_params = search_model(
            X_best,
            y,
            test_size,
            val_size,
            period_type=period_type,
            non_ml_models=True,
            two_models=two_models,
            quick_mod=quick_mod,
        )
        ts_pred = predict(X_best, y, period_type, pred_period, best_model, best_params, refit)
        predictions.append(ts_pred.rename(obj_name))
    predictions = pd.concat(predictions, axis=1)
    predictions.columns = list(predictions.columns)  # transform multiindex to list (for edges)
    if norm:
        predictions = normalize_rows(predictions)
    return predictions


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    # TODO если одна нода попала на выходной, у неё будет вероятность 1, хотя она обычно, например, в районе 0.1 - выброс
    # предложение - не нормировать и предсказывать 'число раз' напрямую
    # можно отнормировать весь df перед ml алгоритмом, предсказать, потом предсказание преобразовать обратно

    return df.div(df.sum(axis=1), axis=0)
