from typing import List, Optional

from numpy import abs as np_abs
from numpy import float as np_float
from pandas import DataFrame, Series, concat, get_dummies, to_datetime, to_numeric
from pandas.api.types import is_numeric_dtype

from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

from shap import LinearExplainer

from process_mining.sberpm._holder import DataHolder


class FactorAnalysis:
    def __init__(
        self,
        data_holder: DataHolder,
        target_column: str,
        type_of_target: str,  # types of target: "number", "string", "time"
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        date_cols: Optional[List[str]] = None,
        extended_search: bool = False,
        count_others: bool = False,
    ) -> None:

        self.target_column = target_column
        self.categorical_cols = [] if categorical_cols is None else categorical_cols
        self.numeric_cols = [] if numeric_cols is None else numeric_cols
        self.date_cols = [] if date_cols is None else date_cols
        self._reset_categorical_cols = self.categorical_cols.copy()
        self._reset_numeric_cols = self.numeric_cols.copy()
        self._reset_date_cols = self.date_cols.copy()
        self.extended_search = extended_search
        self.count_others = count_others
        self.type_of_target = type_of_target
        self._result = DataFrame()

        feature_cols = self.numeric_cols + self.date_cols + self.categorical_cols
        if len(feature_cols) == 0:
            raise ValueError(
                "At least one feature column must be given "
                "(categorical_cols, numeric_cols or/and date_cols arguments)."
            )

        elif target_column in feature_cols:
            raise ValueError("The target column cannot be a feature.")

        if type_of_target not in ["number", "string", "time"]:
            raise ValueError(
                "The type of the target column can take one of the following values: "
                '"number", "string", "time".'
            )

        if target_column == "duration":
            data_holder.check_or_calc_duration()

        self.data = data_holder.data[feature_cols + [target_column]].copy()

    @staticmethod
    def encoding(series: Series) -> DataFrame:
        val_count = series.value_counts(dropna=False).rename(series.name)
        result = DataFrame({"col": series})
        return result.join(val_count, how="left", on="col")[series.name]

    def one_hot(self, df: DataFrame):
        cols_dict = {}

        for col in self.categorical_cols.copy():
            if 1 < df[col].nunique() < 100:
                one_hot = get_dummies(df[col])
                new_col_names = [f"{col}_{value}" for value in one_hot.columns]
                one_hot.columns = new_col_names
                cols_dict[col] = new_col_names
                df = concat([df, one_hot], axis=1).drop(col, axis=1)
            elif df[col].nunique() >= 100:
                print("Too many unique values in", col)
                # convert to numeric type
                if not is_numeric_dtype(df[col]):
                    df[col] = self.encoding(df[col])
                    self.numeric_cols.append(col)
                    self.categorical_cols.remove(col)
                    df[col] = to_numeric(df[col], errors="coerce").fillna(0).astype(np_float)
                    # cleaup (откидывание выбросов)
                    if df[col].nunique() == 1:
                        df = df.drop(col, axis=1)
                        self.numeric_cols.remove(col)

            elif df[col].nunique() == 1:
                print("Too few unique values in", col)
                df = df.drop(col, axis=1)
                self.categorical_cols.remove(col)
            else:
                raise RuntimeError()
        df = df.sort_index(axis=1)

        if df.drop(self.target_column, axis=1).empty:
            raise ValueError("Empty feature list in your data, please check your data")
        return df, cols_dict

    def r2(self, X, y, columns_dictionary) -> Series:
        score = Series([])
        lr = linear_model.Ridge(random_state=42)
        full_score = cross_val_score(lr, X, y, error_score="raise")
        full_score = max(full_score)
        for col in self.categorical_cols + self.numeric_cols:
            if len(self.categorical_cols + self.numeric_cols) == 1:
                cols = X.columns
            elif col in columns_dictionary:
                cols = list(set(X.columns) - set(columns_dictionary[col]))
            else:
                cols = list(set(X.columns) - {col})
            lr = linear_model.Ridge(random_state=42)
            sc = cross_val_score(lr, X[cols], y)
            sc = max(sc)
            score[col] = full_score - sc

        score.name = "r2_Ridge"
        score = score.abs()
        if float(score.sum()) != 0:
            score = score / float(score.sum()) * 100
        return score

    @staticmethod
    def lin_reg(X, y, columns_dictionary):
        lr = linear_model.Ridge(random_state=42)
        lr.fit(X, y)
        res_lr = np_abs(lr.coef_)
        not_explained_score = lr.score(X, y)

        res_lr = DataFrame(res_lr, index=X.columns).transpose()
        for old_col, current_cols in columns_dictionary.items():
            res_lr[old_col] = res_lr[current_cols].pow(2).sum(axis=1).pow(1 / 2) / len(current_cols)
            res_lr = res_lr.drop(current_cols, axis=1)

        if float(res_lr.sum(axis=1)) != 0:
            res_lr = res_lr / float(res_lr.sum(axis=1)) * 100
        res_lr.index = ["lin_reg"]
        return res_lr, not_explained_score

    def shap(self, X_train, y_train, X_test, columns_dictionary):
        lr = linear_model.Lars(random_state=42)
        lr.fit(X_train, y_train)
        explainer = LinearExplainer(model=lr, data=X_train, masker=X_train)
        shap_values = explainer.shap_values(X_test)

        res_shap_tree = DataFrame(shap_values, columns=X_test.columns)
        for old_col, current_cols in columns_dictionary.items():
            res_shap_tree[old_col] = res_shap_tree[current_cols].pow(2).sum(axis=1).pow(1 / 2) / len(current_cols)
            res_shap_tree = res_shap_tree.drop(current_cols, axis=1)

        res_shap_tree = res_shap_tree.iloc[0]
        res_shap_tree = res_shap_tree / float(res_shap_tree.sum()) * 100

        res_shap_tree.name = "Shap_elastic"
        return res_shap_tree

    def permutation(self, X_train, y_train, X_test, y_test, columns_dictionary):
        model = AdaBoostRegressor(random_state=42).fit(X_train, y_train)
        r = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=-1)
        res = r.importances_mean
        res_perm_imp = DataFrame(res).transpose()
        res_perm_imp.columns = X_test.columns
        for old_col, current_cols in columns_dictionary.items():
            res_perm_imp[old_col] = res_perm_imp[current_cols].pow(2).sum(axis=1).pow(1 / 2) / len(current_cols)
            res_perm_imp = res_perm_imp.drop(current_cols, axis=1)

        res_perm_imp = res_perm_imp.iloc[0]
        res_perm_imp = res_perm_imp / res_perm_imp.sum() * 100

        res_perm_imp.name = "Permutation_AdaBoost"
        return res_perm_imp

    # TODO refactor
    def apply(self) -> DataFrame:
        # reset_cols
        self.categorical_cols = self._reset_categorical_cols.copy()
        self.numeric_cols = self._reset_numeric_cols.copy()
        self.date_cols = self._reset_date_cols.copy()

        # preparation of the table of results
        results = DataFrame([], columns=self.categorical_cols + self.numeric_cols + self.date_cols)
        unique_vals_df = self.data.drop(self.target_column, axis=1, inplace=False).nunique().rename("Unique count")
        results = results.append(unique_vals_df)

        df = self.data.copy()

        # target: time
        if self.type_of_target == "time":
            df[self.target_column] = to_datetime(df[self.target_column])
            df[self.target_column] = to_numeric(df[self.target_column], errors="coerce").fillna(0).astype(np_float)

        # target: string
        elif self.type_of_target == "string":
            df[self.target_column] = self.encoding(df[self.target_column])
            df[self.target_column] = to_numeric(df[self.target_column], errors="coerce").fillna(0).astype(np_float)

        # features: date cols
        for col in self.date_cols:
            df[col] = to_datetime(df[col])
            self.numeric_cols.append(col)

        # features: numeric cols
        for col in self.numeric_cols.copy():
            df[col] = to_numeric(df[col], errors="coerce").fillna(0).astype(np_float)
            # cleaup (откидывание выбросов)
            if df[col].nunique() == 1:
                df = df.drop(col, axis=1)
                self.numeric_cols.remove(col)
                print("Too few unique values in", col)

        if (
            df.drop(self.target_column, axis=1).empty
            or len(self.categorical_cols + self.numeric_cols + self.date_cols) == 0
        ):
            raise ValueError(
                "The data turned out to be empty after clearing incorrect values, please check your data"
            )

        # features: categorical cols - one_hot
        data_hot, columns_dictionary = self.one_hot(df)
        data_hot = data_hot.dropna()

        if data_hot.empty:
            raise ValueError(
                "The data turned out to be empty after clearing the empty values, please check your data"
            )

        # Test_train
        data_scaled = DataFrame(MinMaxScaler().fit_transform(data_hot), columns=data_hot.columns)
        X, y = data_scaled.drop(self.target_column, axis=1), data_scaled[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Results
        shap_score = self.shap(X_train, y_train, X_test, columns_dictionary)
        shap_score = shap_score.abs() / shap_score.abs().sum() * 100
        permutation_score = self.permutation(X_train, y_train, X_test, y_test, columns_dictionary)
        permutation_score = permutation_score.abs() / permutation_score.abs().sum() * 100
        if self.count_others:
            res_lr, not_explained_score = self.lin_reg(X, y, columns_dictionary)
            if not_explained_score < 0:
                not_explained_score = 1
            r2_score = self.r2(X, y, columns_dictionary)
            if any(r2_score <= 0):
                r2_coef = 0
            elif self.extended_search:
                r2_coef = 9
            else:
                r2_coef = 5

            results = results.append(not_explained_score * r2_score)
            results = results.append(not_explained_score * res_lr)
            results = results.append(not_explained_score * shap_score)
            if self.extended_search:
                results = results.append(not_explained_score * permutation_score)
            results["Прочее"] = 100 * (1 - not_explained_score)
        else:
            r2_score = self.r2(X, y, columns_dictionary)
            results = results.append(r2_score)

            if any(r2_score <= 0):
                r2_coef = 0
            elif self.extended_search:
                r2_coef = 9
            else:
                r2_coef = 5
            results = results.append(self.lin_reg(X, y, columns_dictionary)[0])
            results = results.append(shap_score)
            if self.extended_search:
                results = results.append(permutation_score)

        results = results.fillna(0)
        res = results.copy()
        res = res.append(
            Series(
                (r2_coef * res.loc["r2_Ridge"] + res.loc["lin_reg"] + 4 * res.loc["Shap_elastic"]) / (r2_coef + 5),
                name="results_default",
            )
        )
        if self.extended_search:
            res = res.append(
                Series(
                    (
                        r2_coef * results.loc["r2_Ridge"]
                        + results.loc["lin_reg"]
                        + 4 * results.loc["Shap_elastic"]
                        + 4 * results.loc["Permutation_AdaBoost"]
                    )
                    / (r2_coef + 9),
                    name="results_extended",
                )
            )

        self._result = res.tail(1).copy()
        return self._result.copy()

    def get_result(self) -> DataFrame:
        if not self._result.empty:
            return self._result.copy()
        else:
            raise RuntimeError("Call apply() method for FactorAnalysis object first.")
