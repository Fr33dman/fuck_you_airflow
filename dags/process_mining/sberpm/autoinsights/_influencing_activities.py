from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from numpy import dot, log, mean
from numpy import sum as np_sum
from pandas import DataFrame, concat, pivot_table

from catboost import CatBoostRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from shap import Explainer, KernelExplainer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    Ridge,
    TheilSenRegressor,
)
from sklearn.model_selection import GridSearchCV

from tqdm.notebook import tqdm

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.metrics import IdMetric


class Influencing_activities:
    def __init__(
        self,
        data: DataHolder,
        mode: str = "best",
        metric: str = "appearance",
        metric_f: str = "appearance",  # 'appearance', 'recycles', 'time', 'user_metric'
        user_metric_column: str = None,
        user_metric_column_f: str = None,
        auto_i: bool = False,
    ) -> None:

        self._dh = data

        if mode not in ["best", "fast"]:
            raise ValueError(
                f"The 'mode' parameter received the value '{mode}'. Expected value of 'best' or 'fast'"
            )

        self._mode = mode

        if metric not in ["appearance", "recycles", "time", "user_metric"]:
            raise ValueError(
                f"The 'metric' parameter received the value '{metric}'. Expected value of 'appearance', "
                f"'recycles', 'user_metric' or 'fast'."
            )

        if metric_f not in ["appearance", "recycles", "time", "user_metric"]:
            raise ValueError(
                f"The 'metric_f' parameter received the value '{metric_f}'. Expected value of 'appearance', "
                f"'recycles', 'user_metric' or 'fast'."
            )

        self._metric = metric
        self._metric_f = metric_f
        self._user_metric_column = user_metric_column
        self._user_metric_column_f = user_metric_column_f

        if self._metric == "user_metric" and user_metric_column is None:
            raise ValueError("To use use_metric, you need to specify user_metric_column.")
        if self._metric_f == "user_metric" and user_metric_column_f is None:
            raise ValueError("To use use_metric, you need to specify user_metric_column.")

        self._models_dict = {
            "RandomForestRegressor": {
                "params": {
                    "n_estimators": [20],
                    "max_depth": [2, 3, 4],
                    "max_features": ["sqrt", "log2", 0.5],
                    "min_samples_leaf": [1, 2, 4],
                    "random_state": [420],
                },
                "model": RandomForestRegressor(),
            },
            "HuberRegressor": {
                "params": {"epsilon": [1.35, 1.5], "alpha": [1e-08, 1e-04, 1e-02, 1e-01, 5e-01, 1]},
                "model": HuberRegressor(),
            },
            "GradientBoostingRegressor": {
                "params": {
                    "loss": ["squared_error", "lad", "huber", "quantile"],
                    "n_estimators": [10, 20, 50],
                    "learning_rate": [0.01, 0.1, 0.5],
                    "max_depth": [2, 3, 4],
                    "random_state": [420],
                },
                "model": GradientBoostingRegressor(),
            },
            "PassiveAggressiveRegressor": {
                "params": {
                    "C": [1, 100, 0.01, 1000, 0.001, 0.1, 2, 0.5],
                    "epsilon": [0.01, 0.05, 0.1, 0.5],
                    "validation_fraction": [0.05, 0.1, 0.2, 0.5],
                    "fit_intercept": [True, False],
                    "random_state": [420],
                },
                "model": PassiveAggressiveRegressor(),
            },
            "ElasticNet": {
                "params": {
                    "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    "alpha": [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
                    "fit_intercept": [True, False],
                    "random_state": [420],
                },
                "model": ElasticNet(),
            },
            "Ridge": {
                "params": {
                    "alpha": [1e-08, 1e-04, 1e-02, 0.01, 0.1, 0.2, 0.5],
                    "fit_intercept": [True, False],
                    "random_state": [420],
                },
                "model": Ridge(),
            },
            "TheilSenRegressor": {"params": {"random_state": [420]}, "model": TheilSenRegressor()},
            "OrthogonalMatchingPursuit": {"params": {}, "model": OrthogonalMatchingPursuit()},
        }

        self._scoring_list = {"R2": "r2", "MAE": "neg_mean_absolute_error"}
        self._result = DataFrame()
        self._search_model_res = list()
        self._final_search_res = list()

        self._auto_i = auto_i

    def _search_model(self, shape, df: DataFrame, pred_column=0, features=None):
        best_error_metric = 0.0
        best_model = "No model"
        params = 0
        data1 = df.iloc[:shape, :]
        data2 = df.iloc[-shape:, :]

        if self._mode == "best":
            all_models_together = [
                "HuberRegressor",
                "Ridge",
                "TheilSenRegressor",
                "GradientBoostingRegressor",
                "PassiveAggressiveRegressor",
                "RandomForestRegressor",
                "ElasticNet",
                "OrthogonalMatchingPursuit",
            ]

            val = 3
            for model1 in tqdm(all_models_together):
                try:
                    cv = GridSearchCV(
                        estimator=self._models_dict[model1]["model"],
                        param_grid=self._models_dict[model1]["params"],
                        n_jobs=-1,
                        cv=val,
                        refit=True,
                        scoring="r2",
                    )
                    cv.fit(data2.loc[:, features].to_numpy(), data1.iloc[:, pred_column].to_numpy())
                    if cv.best_score_ > best_error_metric:
                        params = cv.best_params_
                        best_error_metric = cv.best_score_
                        best_model = model1
                except Exception as error:  # TODO localize error
                    # TODO test with loguru
                    print(error)
                    print("Continue to search")

            # на случай, если ни одна модель не отработала лучше 0
            if best_model == "No model":
                best_model = "GradientBoostingRegressor"

        elif self._mode == "fast":
            best_error_metric, best_model, params, _ = None, "ElasticNet", None, None  # temp

        self._search_model_res = {
            "Best error_metric": best_error_metric,
            "Best model": best_model,
            "Best params": params,
        }

        return self._search_model_res

    def _final_search(self, shape, df, pred_column=0, features=None, model=None):
        val = 3

        par = []
        err = []
        models = []
        mae_err = []
        data1 = df.iloc[:shape, :]
        data2 = df.iloc[-shape:, :]

        if self._mode == "best":
            for k in tqdm(range(len(data2.columns))):
                try:
                    cv = GridSearchCV(
                        estimator=self._models_dict[model]["model"],
                        param_grid=self._models_dict[model]["params"],
                        n_jobs=-1,
                        cv=val,
                        refit="R2",
                        scoring=self._scoring_list,
                    )
                    cv.fit(data2.loc[:, features[k]].to_numpy(), data1.iloc[:, k].to_numpy())
                    par.append(cv.best_params_)
                    err.append(cv.best_score_)
                    mae_err.append(cv.cv_results_["mean_test_MAE"][cv.best_index_])
                    models.append(cv)
                except Exception as error:  # TODO localize error
                    # TODO test with loguru
                    print("Error")
                    print(error)
                    break

            if (mean(err) < 0.0) & (model != "GradientBoostingRegressor"):
                SM = self._final_search(
                    shape=shape,
                    df=df,
                    pred_column=pred_column,
                    features=features,
                    model="GradientBoostingRegressor",
                )
                err, models, par, mae_err = SM["Best error_metric"], SM["models"], SM["Best params"], SM["mae"]

        elif self._mode == "fast":
            data1 = df.iloc[:shape, :]
            data2 = df.iloc[-shape:, :]
            try:
                model = "ElasticNet"
                for k in tqdm(range(len(data2.columns))):
                    cv = GridSearchCV(
                        estimator=self._models_dict[model]["model"],
                        param_grid=self._models_dict[model]["params"],
                        n_jobs=-1,
                        cv=val,
                        refit="R2",
                        scoring=self._scoring_list,
                    )
                    cv.fit(data2.loc[:, features[k]].to_numpy(), data1.iloc[:, k].to_numpy())
                    par.append(cv.best_params_)
                    err.append(cv.best_score_)
                    mae_err.append(cv.cv_results_["mean_test_MAE"][cv.best_index_])
                    models.append(cv)
            except Exception:  # TODO localize error
                # TODO test with loguru
                model = "Ridge"
                for k in tqdm(range(len(data2.columns))):
                    cv = GridSearchCV(
                        estimator=self._models_dict[model]["model"],
                        param_grid=self._models_dict[model]["params"],
                        n_jobs=-1,
                        cv=val,
                        refit="R2",
                        scoring=self._scoring_list,
                    )
                    cv.fit(data2.loc[:, features[k]].to_numpy(), data1.iloc[:, k].to_numpy())
                    par.append(cv.best_params_)
                    err.append(cv.best_score_)
                    mae_err.append(cv.cv_results_["mean_test_MAE"][cv.best_index_])
                    models.append(cv)

        self._final_search_res = {"Best error_metric": err, "models": models, "Best params": par, "mae": mae_err}

        return self._final_search_res

    def retrieve_from_fitting(self, data, data1, data2, k, val):
        X, y = data2.loc[:, [i for i in data2.columns if i != k]], data1.loc[:, k]

        sfs_cv = SFS(
            LinearRegression(),
            k_features="parsimonious",
            forward=True,
            floating=True,
            verbose=0,
            scoring="r2",
            cv=val,
            n_jobs=-1,
        )
        sfs_cv = sfs_cv.fit(X, y)
        name = list(sfs_cv.k_feature_names_)

        if len(name) < 2:
            sfs_cv = SFS(
                LinearRegression(),
                k_features="best",
                forward=True,
                floating=True,
                verbose=0,
                scoring="r2",
                cv=val,
                n_jobs=-1,
            )
            sfs_cv = sfs_cv.fit(X, y)
            name = [i for i in data.columns if i != k]

        return [len(sfs_cv.k_feature_names_), sfs_cv.k_score_, name, k]

    def _features_selection(self, data, shape):
        data1 = data.iloc[:shape, :]
        data2 = data.iloc[-shape:, :]

        if self._mode == "best":
            val = 4

            with ThreadPoolExecutor() as executor:
                leng = executor.map(
                    self.retrieve_from_fitting,
                    repeat(data),
                    repeat(data1),
                    repeat(data2),
                    list(data.columns),
                    repeat(val),
                )

            x = DataFrame(leng)
            try:
                ind = x[(x.iloc[:, 0] >= 0.1 * x.shape[0]) & (x.iloc[:, 1] < 1.0)].iloc[:, 1].idxmax()
            except Exception:
                # TODO test with loguru
                try:
                    ind = x[(x.iloc[:, 0] >= 0.1 * x.shape[0])].iloc[:, 1].idxmax()
                except Exception:  # TODO chain exceptions
                    ind = x.iloc[:, 1].idxmax()
            cat_names = []
            X, y = data2.iloc[:, [i != ind for i in range(len(data2.columns))]], data1.iloc[:, ind]

            sfs_cv = SFS(
                CatBoostRegressor(
                    iterations=50, eval_metric="R2", silent=True, random_state=420, allow_writing_files=False
                ),
                k_features="parsimonious",
                forward=True,
                floating=False,
                verbose=0,
                scoring="r2",
                cv=val,
                n_jobs=-1,
            )
            try:
                sfs_cv = sfs_cv.fit(X, y)
                cat_names = sfs_cv.k_feature_names_
            except Exception:  # TODO else finally
                print("just Exception")
                # TODO test with loguru
            united_features = list(set(cat_names).union(set(x.iloc[ind][2])))

            feat = list(x.iloc[:, 2])

        elif self._mode == "fast":
            united_features, ind, feat = None, None, [[i for i in data2.columns if i != k] for k in data2.columns]

        return {"good_features": united_features, "index": ind, "selected_features": feat}

    def _get_activity_poblematicity(self, data, features, models, shape):
        data1 = data.iloc[:shape, :]
        data2 = data.iloc[-shape:, :]
        mean_val = data1.mean(axis=0)
        ser = DataFrame({"Features": data2.columns})
        for k in range(len(data2.columns)):

            model = models["models"][k].best_estimator_
            try:
                explainer = Explainer(model, data2[features[k]], check_additivity=False)
            except Exception as err:
                # TODO test with loguru
                print(err)
                explainer = KernelExplainer(model.predict, data2[features[k]], check_additivity=False)
            try:
                shap_values = explainer(data2[features[k]])
            except Exception as err:
                print(err)
                shap_values = explainer(data2[features[k]], check_additivity=False)
            df = DataFrame({"coefs": shap_values[0].values, "Features": data2[features[k]].columns})
            ser = ser.merge(df, how="left", on="Features")
            ser.iloc[k, -1] = mean_val.iloc[k]

        ser = ser.set_index("Features")
        if self._metric == "time":
            data3 = data.iloc[shape:, :].iloc[:-shape, :]
            result = DataFrame(
                {
                    "Poblematicity": dot(
                        ser.fillna(0).to_numpy(),
                        data3.sum(axis=0).apply(lambda x: x / data3.sum(axis=0).sum()).to_numpy(),
                    ),
                    "Weight in dataset": data3.sum(axis=0).apply(lambda x: x / data3.sum(axis=0).sum()),
                    "Selected features": features,
                }
            ).sort_values(by="Poblematicity", ascending=False)
        else:
            result = DataFrame(
                {
                    "Poblematicity": dot(
                        ser.fillna(0).to_numpy(),
                        data1.sum(axis=0).apply(lambda x: x / data1.sum(axis=0).sum()).to_numpy(),
                    ),
                    "Weight in dataset": data1.sum(axis=0).apply(lambda x: x / data1.sum(axis=0).sum()),
                    "Selected features": features,
                }
            ).sort_values(by="Poblematicity", ascending=False)

        result["Poblematicity"] = result["Poblematicity"] / result["Poblematicity"].apply(abs).max()
        return result

    def _extract_data(self, data_holder: DataHolder, user_metric_column: str):

        if data_holder.start_timestamp_column and len(data_holder.start_timestamp_column) > 0:
            a = data_holder.start_timestamp_column
        elif data_holder.end_timestamp_column and len(data_holder.end_timestamp_column) > 0:
            a = data_holder.end_timestamp_column

        if self._metric_f == "appearance" or self._metric_f != "time" and self._metric_f == "recycles":
            data_f = pivot_table(
                data_holder.data,
                values=a,
                index=data_holder.id_column,
                columns=data_holder.activity_column,
                aggfunc="count",
            ).fillna(0)

        elif self._metric_f == "time":
            data_holder.check_or_calc_duration()
            data_holder.data["duration"] = data_holder.data["duration"].fillna(0)
            idt = data_holder.data[data_holder.data["duration"] < 0][data_holder.id_column]
            data_holder.data = data_holder.data[~data_holder.data[data_holder.id_column].isin(idt)]

            data_f = (
                pivot_table(
                    data_holder.data,
                    values="duration",
                    index=data_holder.id_column,
                    columns=data_holder.activity_column,
                    aggfunc=np_sum,
                )
                .fillna(0)
                .applymap(lambda x: log(x + 1))
            )

        elif self._metric_f == "user_metric_f":
            data_f = pivot_table(
                data_holder.data,
                values=user_metric_column,
                index=data_holder.id_column,
                columns=data_holder.activity_column,
                aggfunc=np_sum,
            ).fillna(0)

        if self._metric == "appearance":
            data_t = pivot_table(
                data_holder.data,
                values=a,
                index=data_holder.id_column,
                columns=data_holder.activity_column,
                aggfunc="count",
            ).fillna(0)

        elif self._metric == "time":
            data_holder.check_or_calc_duration()
            data_holder.data["duration"] = data_holder.data["duration"].fillna(0)
            id_metric = IdMetric(data_holder, time_unit="s")
            idt = data_holder.data[data_holder.data["duration"] < 0][data_holder.id_column]
            data_holder.data = data_holder.data[~data_holder.data[data_holder.id_column].isin(idt)]

            data_r = pivot_table(
                data_holder.data,
                values="duration",
                index=data_holder.id_column,
                columns=data_holder.activity_column,
                aggfunc=np_sum,
            ).fillna(0)
            data_log = data_r.applymap(lambda x: log(x + 1))
            data_t = concat([data_log, data_r], axis=0)

        elif self._metric == "recycles":
            data_t = (
                pivot_table(
                    data_holder.data,
                    values=a,
                    index=data_holder.id_column,
                    columns=data_holder.activity_column,
                    aggfunc="count",
                )
                .fillna(0)
                .applymap(lambda x: x - 1 if x > 0 else 0)
            )

        elif self._metric == "user_metric_t":
            data_t = pivot_table(
                data_holder.data,
                values=user_metric_column,
                index=data_holder.id_column,
                columns=data_holder.activity_column,
                aggfunc=np_sum,
            ).fillna(0)

        data = concat([data_t, data_f], axis=0)

        return {"data": data, "metric": self._metric, "metric_f": self._metric_f, "shape": data_f.shape[0]}

    def activities_impact(self, feature_counts=True):
        extracted_data = self._extract_data(
            self._dh, user_metric_column=self._user_metric_column
        )

        features = list(extracted_data["data"].columns)

        extracted_data["data"] = extracted_data["data"].astype("float")
        res = self._features_selection(
            extracted_data["data"], shape=extracted_data["shape"]
        )

        if feature_counts:
            features_appearance = sum(res["selected_features"], [])
            return {feature: features_appearance.count(feature) for feature in features}

        return dict(zip(features, res["selected_features"]))
