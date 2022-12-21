from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_selection import f_regression

from process_mining.sberpm.ml.graph_structure_prediction import train_test_val_split


def mape(y_true, y_pred):  # MAPE scoring
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    val_size: float,
    best_model: Union[BaseEstimator, List[BaseEstimator]],
    res_feature_share: float = 0.3,
) -> List[int]:
    """
    На вход берет , датасет факта, номер столбца в датасете (по умолчанию 0),
    количество предсказываемых месяцев и набор моделей для проверки.

    Гридсерч по гиперпараметрам по модели по MAPE

    На выходе получается list лучших фичей и list их MAPE

    """
    assert best_model is not None

    results = {}

    X_train, X_test, _, y_train, y_test, _ = train_test_val_split(X, y, train_size, val_size)
    X_y_train_test = [X_train, X_test, y_train, y_test]

    num_features = int(X.shape[1] * res_feature_share)

    results["SequentialFeatureSelector"] = sequential_feature_selector(X_y_train_test, best_model, num_features)

    results["SelectKBest"] = select_k_best(X_y_train_test, best_model, num_features)

    # у модели должны быть аттрибуты: feature_importances_ или coef_
    if are_feature_importance_supported(best_model):
        results["SelectFromModel"] = select_from_model(X_y_train_test, best_model, num_features)

    best_indexes, best_score = min(results.values(), key=lambda x: x[1])  # min score

    return best_indexes


def sequential_feature_selector(X_y_train_test: list, best_model, num_features: int) -> Tuple[list, float]:
    scorer = sklearn.metrics.make_scorer(mape, greater_is_better=False)
    X_train, X_test, y_train, y_test = X_y_train_test
    if not isinstance(best_model, list):
        sfs = SequentialFeatureSelector(best_model, k_features=num_features, scoring=scorer, clone_estimator=True)
        sfs.fit(X_train, y_train)
        indexes = list(sfs.k_feature_idx_)

        best_model_clone = clone(best_model)
        X_train_tr = sfs.transform(X_train)
        X_test_tr = sfs.transform(X_test)
        best_model_clone.fit(X_train_tr, y_train)
        y_pred = best_model_clone.predict(X_test_tr)
        score = mape(y_test, y_pred)
    else:
        sfs0 = SequentialFeatureSelector(
            best_model[0], k_features=num_features, scoring=scorer, clone_estimator=True
        )
        sfs0.fit(X_train, y_train)
        sfs1 = SequentialFeatureSelector(
            best_model[1], k_features=num_features, scoring=scorer, clone_estimator=True
        )
        sfs1.fit(X_train, y_train)
        indexes = list(set(sfs0.k_feature_idx_) | set(sfs1.k_feature_idx_))

        best_model_clone = [clone(m) for m in best_model]
        X_train_tr = X_train.iloc[:, indexes]
        X_test_tr = X_test.iloc[:, indexes]
        best_model_clone[0].fit(X_train_tr, y_train)
        y_pred0 = best_model_clone[0].predict(X_test_tr)
        best_model_clone[1].fit(X_train_tr, y_train)
        y_pred1 = best_model_clone[1].predict(X_test_tr)
        y_pred = ((y_pred0 + y_pred1) / 2 + np.sqrt(y_pred0 * y_pred1)) / 2
        score = mape(y_test, y_pred)
    return indexes, score


def select_k_best(X_y_train_test: list, best_model, num_features: int) -> Tuple[list, float]:
    X_train, X_test, y_train, y_test = X_y_train_test
    skb = SelectKBest(score_func=f_regression, k=num_features)
    skb.fit(X_train, y_train)
    indexes = list(skb.get_support(indices=True))

    X_train_tr = skb.transform(X_train)
    X_test_tr = skb.transform(X_test)

    if not isinstance(best_model, list):
        best_model_clone = clone(best_model)
        best_model_clone.fit(X_train_tr, y_train)
        y_pred = best_model_clone.predict(X_test_tr)
        score = mape(y_test, y_pred)
    else:
        best_model_clone = [clone(m) for m in best_model]
        best_model_clone[0].fit(X_train_tr, y_train)
        y_pred0 = best_model_clone[0].predict(X_test_tr)
        best_model_clone[1].fit(X_train_tr, y_train)
        y_pred1 = best_model_clone[1].predict(X_test_tr)
        y_pred = ((y_pred0 + y_pred1) / 2 + np.sqrt(y_pred0 * y_pred1)) / 2
        score = mape(y_test, y_pred)

    return indexes, score


def select_from_model(X_y_train_test: list, best_model, num_features: int) -> Tuple[list, float]:
    X_train, X_test, y_train, y_test = X_y_train_test
    sfm = SelectFromModel(best_model, max_features=num_features)
    sfm.fit(X_train, y_train)
    X_train_tr = sfm.transform(X_train)
    X_test_tr = sfm.transform(X_test)
    indexes = list(sfm.get_support(indices=True))
    best_model.fit(X_train_tr, y_train)
    y_pred = best_model.predict(X_test_tr)
    score = mape(y_test, y_pred)
    return indexes, score


def are_feature_importance_supported(model: object) -> bool:
    if not isinstance(model, list):
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):  # models are fitted
            return True
    else:
        # True if both models have needed attributes
        return np.all([hasattr(m, "feature_importances_") or hasattr(m, "coef_") for m in model])
