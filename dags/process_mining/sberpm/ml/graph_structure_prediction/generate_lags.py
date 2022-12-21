from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from process_mining.sberpm.ml.graph_structure_prediction import working_days


def generate_lags(ts: pd.Series, period_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    num_lags = get_num_points(period_type)

    # Generate lags columns and set correct indexes
    features = pd.DataFrame({"y": ts})
    for lag_num in range(1, num_lags + 1)[::-1]:
        features[f"lag_{lag_num}"] = features["y"].shift(lag_num)

    features = features.iloc[num_lags:]  # remove first num_lags elements (None values)
    y = features["y"]
    features = features.drop("y", axis=1)

    # Generate diff columns
    for lag_num in range(2, num_lags + 1):
        features[f"diff_{1}-{lag_num}"] = features[f"lag_{1}"] - features[f"lag_{lag_num}"]

    # Generate one-hot encoded month columns
    if period_type != "Y":
        features_months = pd.DataFrame(
            OneHotEncoder()
            .fit(np.arange(1, 13).reshape(-1, 1))  # 12 months
            .transform(features.index.month.values.reshape(-1, 1))
            .toarray(),
            columns=[f"M{i}" for i in range(1, 13)],
            index=features.index,
        )
        features = features.join(features_months)

    # Generate working days columns
    if period_type == "M":
        wd = working_days().set_index("date")
        if features.index.min() >= wd.index.min() and features.index.max() <= wd.index.max():
            features = features.join(wd["month_days"])
        # features = pd.merge(features, work_days.set_index('date')['month_days'],
        #                     how='left', left_index=True, right_index=True)

    # Generale linear column
    features["linear"] = np.arange(1, features.shape[0] + 1)

    return features, y


def generate_one_lag(
    ts: pd.Series, ts_pred_index: pd.DatetimeIndex, ts_pred_values: List[float], period_type: str
) -> pd.DataFrame:
    """
    Generate features for one element.
    """
    num_lags = get_num_points(period_type)
    ts_pred = pd.Series(ts_pred_values, index=ts_pred_index[: len(ts_pred_values)], dtype="float")
    pred_point_date = ts_pred_index[len(ts_pred_values) : len(ts_pred_values) + 1]  # pd.Series, len=1

    # lags
    cols = [f"lag_{i}" for i in range(num_lags, 0, -1)]
    features = list(pd.concat([ts[len(ts) - num_lags + len(ts_pred) :], ts_pred[-num_lags:]], axis=0))

    # diff
    cols += [f"diff_{1}-{i}" for i in range(2, num_lags + 1)]
    features += list(map(lambda x: features[-1] - x, features[:-1][::-1]))

    # one hot
    if period_type != "Y":
        cols += [f"M{i}" for i in range(1, 13)]
        features += list(
            OneHotEncoder()
            .fit(np.arange(1, 13).reshape(-1, 1))  # 12 months
            .transform(pred_point_date.month.values.reshape(-1, 1))
            .toarray()[0]
        )

    # working days
    if period_type == "M":
        wd = working_days().set_index("date")["month_days"]
        if ts.index[num_lags:].min() >= wd.index.min() and ts.index.max() <= wd.index.max():
            cols.append("month_days")
            features.append(pd.DataFrame(index=pred_point_date).join(wd).values[0][0])

    # linear
    cols.append("linear")
    features.append(len(ts) - num_lags + len(ts_pred) + 1)

    return pd.DataFrame([features], columns=cols)


def train_test_val_split(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float):
    assert 0 < test_size < 1
    assert 0 < val_size < 1
    assert 0 < test_size + val_size < 1
    assert len(X) == len(y)
    train_size = 1 - (test_size + val_size)
    first_test_ind = int(len(y) * train_size)
    first_val_ind = int(len(y) * (train_size + test_size))

    if (X.iloc[first_test_ind:first_val_ind].empty or X.iloc[first_val_ind:].empty) and len(y) > 2:
        first_test_ind = len(y) - 2
        first_val_ind = len(y) - 1

    return (
        X.iloc[:first_test_ind],
        X.iloc[first_test_ind:first_val_ind],
        X.iloc[first_val_ind:],
        y[:first_test_ind],
        y[first_test_ind:first_val_ind],
        y[first_val_ind:],
    )


def get_num_points(period_type: str) -> int:
    assert period_type in ["D", "M", "Y"]
    return {"D": 7, "M": 12, "Y": 5}[period_type]
