import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from process_mining.sberpm.ml.graph_structure_prediction import (
    train_test_val_split,
    get_num_points,
    generate_one_lag,
)
from process_mining.sberpm.ml.graph_structure_prediction.parameters import params_search, estim

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def mape(y_true, y_pred):  # MAPE scoring
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(abs((y_true - y_pred) / y_true)) * 100


def search_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    period_type: str,
    non_ml_models: bool,
    two_models: bool = False,
    quick_mod: bool = False,
) -> Tuple[BaseEstimator, Optional[dict]]:

    if quick_mod:
        all_models = ["AdaboostRegressor", "ElasticNet"]  # TODO AdaboostRegressor, TheilSenRegressor
    else:
        all_models = [
            "DecisionTreeRegressor",
            "HuberRegressor",
            "TheilSenRegressor",
            "GradientBoostingRegressor",
            "PassiveAggressiveRegressor",
            "RandomForestRegressor",
            "KNeighborsRegressor",
            "BaggingRegressor",
            "ElasticNet",
            "AdaboostRegressor",
            "OrthogonalMatchingPursuit",
        ]

    score_func = mse if y.min() <= 1e-09 else mape

    if not non_ml_models:
        X_train, X_test, _, y_train, y_test, _ = train_test_val_split(X, y, test_size, val_size)
    else:
        X_train1, X_train2, X_test, y_train1, y_train2, y_test = train_test_val_split(X, y, test_size, val_size)
        X_train = pd.concat([X_train1, X_train2], axis=0)
        y_train = pd.concat([y_train1, y_train2], axis=0)

    best_score = 10**20
    best_model = None
    best_params = None

    for model1 in all_models:
        cv = GridSearchCV(
            estimator=estim(model1),
            param_grid=params_search(model1),
            n_jobs=-1,
            cv=5,
            refit=True,
            scoring="neg_mean_squared_error",
        )
        cv.fit(X_train, y_train.to_numpy())
        estimator1 = cv.best_estimator_
        y_pred1 = cv.predict(X_test)
        score1 = score_func(y_test, y_pred1)
        if score1 < best_score:
            best_score = score1
            best_model = estimator1
        if two_models is True:
            for model2 in all_models:
                if model2 != model1:
                    cv = GridSearchCV(
                        estimator=estim(model2),
                        param_grid=params_search(model2),
                        n_jobs=-1,
                        cv=5,
                        refit=True,
                        scoring="neg_mean_squared_error",
                    )
                    cv.fit(X_train, y_train)
                    estimator2 = cv.best_estimator_
                    y_pred2 = cv.predict(X_test)

                    y_pred_both = combine_two_preds(y_pred1, y_pred2)
                    score_both = score_func(y_test, y_pred_both)
                    if score_both < best_score:
                        best_score = score_both
                        best_model = [estimator1, estimator2]

    # на случай, если ни одна модель не отработала лучше 10**20 по mape
    if best_model is None and non_ml_models is False:
        best_model = ElasticNet()

    # Additional models
    if non_ml_models is True:
        # Sarimax
        seasonal_order = (0, 0, 0, get_num_points(period_type))
        for tr in ["n", "c", "t", "ct"]:
            model = SARIMAX(y_train, seasonal_order=seasonal_order, trend=tr)
            res = model.fit()
            y_pred = res.forecast(len(y_test))
            score = score_func(y_test, y_pred)
            if score < best_score:
                best_model = model
                best_score = score
                best_params = {"seasonal_order": seasonal_order, "trend": tr}

        # Holt
        model = Holt(y_train, damped_trend=False)
        res = model.fit()
        y_pred = res.forecast(len(y_test))
        score = score_func(y_test, y_pred)
        if score < best_score:
            best_model = model
            best_params = {"damped_trend": False}
            # best_score = score

    return best_model, best_params


def predict(
    X: pd.DataFrame, y: pd.Series, period_type: str, pred_period: int, best_model, best_params, refit: bool
) -> pd.Series:
    if isinstance(best_model, (SARIMAX, Holt)):
        assert best_params is not None
        model = best_model.__class__(y, **best_params)
        if not refit:
            return model.fit().forecast(pred_period)  # pd.Series with proper indexes
        else:
            y_train = y.copy()
            y_pred = []
            for i in range(pred_period):
                y_pred_one = model.fit().forecast(1)
                y_pred.append(y_pred_one)
                y_train = pd.concat([y_train, y_pred_one])
                model = best_model.__class__(y_train, **best_params)
            return pd.concat(y_pred)  # pd.Series with proper indexes
    else:  # one ml model or list of ml models, no need to set params
        models = best_model if isinstance(best_model, list) else [best_model]
        for model in models:
            model.fit(X, y)

        y_pred_index = pd.date_range(y.index.max(), periods=pred_period + 1, freq=y.index.freq)[1:]
        y_pred_values = []
        if refit:
            X_all = X.copy()
            y_all = y.copy()
        for i in range(pred_period):
            obj = generate_one_lag(y, y_pred_index, y_pred_values, period_type)
            obj = obj[X.columns]
            if len(models) == 1:
                pred = max(models[0].predict(obj)[0], 0)
            elif len(models) == 2:
                pred = combine_two_preds(max(models[0].predict(obj)[0], 0), max(models[1].predict(obj)[0], 0))
            else:
                raise RuntimeError()
            y_pred_values.append(pred)

            if refit and i != pred_period - 1:
                X_all = pd.concat([X_all, obj], axis=0)
                y_all = pd.concat([y_all, pd.Series([pred], index=y_pred_index[i : i + 1])], axis=0)
                for model in models:
                    model.fit(X_all, y_all)
        y_pred = pd.Series(y_pred_values, index=y_pred_index)

    return y_pred


def combine_two_preds(y_pred1, y_pred2):
    assert type(y_pred1) == type(y_pred2) == float or type(y_pred1) == type(y_pred2) == np.ndarray
    return ((y_pred1 + y_pred2) / 2 + np.sqrt(y_pred1 + y_pred2)) / 2
