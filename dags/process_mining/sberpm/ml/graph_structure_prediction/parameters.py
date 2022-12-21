from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def params_search(model):
    if "AdaBoostRegressor" in model:
        PARAMS = {"n_estimators": [10, 20, 50], "loss": ["linear", "square", "exponential"]}
    elif "RandomForestRegressor" in model:
        PARAMS = {
            "n_estimators": [20],
            "max_depth": [2, 3, 4],
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_leaf": [1, 2, 4],
        }
    # elif model == 'ElasticNet':
    #    PARAMS = {
    #      'l1_ratio': [0.1, 0.2, 0.4, 0.8],
    #      'alpha': [1e-06, 1e-05, 1e-04, 1e-03]
    #     }
    elif "ExtraTreesRegressor" in model:
        PARAMS = {
            "n_estimators": [20],
            "max_depth": [2, 3, 4],
            "max_features": ["sqrt", "log2", 0.5, 1.0],
            "min_samples_leaf": [1, 2, 4],
        }
    elif "BaggingRegressor" in model:
        PARAMS = {"n_estimators": [20], "max_samples": [0.5], "max_features": [0.5]}
    elif "HuberRegressor" in model:
        PARAMS = {"alpha": [1e-08, 1e-04, 1e-02, 1e-01, 5e-01, 1]}
    elif "Lasso" in model:
        PARAMS = {
            "alpha": [1e-08, 1e-04, 1e-02, 1e-01, 2e-01, 5e-01, 1, 2, 10, 100],
            "fit_intercept": [True, False],
        }
    elif "LassoLars" in model:
        PARAMS = {"alpha": [1e-08, 1e-04, 1e-02, 1e-01]}
    elif "GradientBoostingRegressor" in model:
        PARAMS = {
            "loss": ["ls", "lad", "huber", "quantile"],
            "n_estimators": [10, 20, 50],
            "learning_rate": [0.01, 0.1, 0.5],
            "max_depth": [2, 3, 4],
        }
    elif "DecisionTreeRegressor" in model:
        PARAMS = {
            "min_samples_leaf": [1, 2, 4, 8],
            "max_depth": [1, 2, 4, 8],
        }
    elif "MLPRegressor" in model:
        PARAMS = {
            "hidden_layer_sizes": [10, 100],
            # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
            "activation": ["logistic", "tanh"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.00001, 0.0001, 0.001],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "beta_1": [0.8, 0.9, 0.99],
            "beta_2": [0.99, 0.999, 0.9999],
        }

    elif "PassiveAggressiveRegressor" in model:
        PARAMS = {
            "C": [1, 100, 0.01, 1000, 0.001, 0.1, 2, 0.5],
            "epsilon": [0.01, 0.05, 0.1, 0.5],
            "validation_fraction": [0.05, 0.1, 0.2, 0.5],
            "fit_intercept": [True, False],
        }
    elif "ElasticNet" in model:
        PARAMS = {
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "alpha": [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
            "fit_intercept": [True, False],
        }
    elif "BayesianRidge" in model:
        PARAMS = {"alpha_1": [1e-08, 1e-04, 1e-02, 0.01]}
    elif "Ridge" in model:
        PARAMS = {"alpha": [1e-08, 1e-04, 1e-02, 0.01, 0.1, 0.2, 0.5], "fit_intercept": [True, False]}
    elif "SGD" in model:
        PARAMS = {"alpha": [1e-08, 1e-04, 1e-02, 1e-01]}
    elif "SVR" in model:
        PARAMS = {"kernel": ["poly"], "degree": [1, 2, 3]}
    elif "KNeighbors" in model:
        PARAMS = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7], "p": [1, 2]}
    elif "LGBMRegressor" in model:
        PARAMS = {
            "feature_fraction": [0.5, 0.8],
            "n_estimators": [50],
            "learning_rate": [0.001, 0.01, 0.1],
            "min_child_samples": [2, 4],
            "num_leaves": [16],
        }
    elif "Holt" in model:
        PARAMS = {"damped": [True, False], "exponential": [True, False]}
    elif "SARIMAX" in model:
        PARAMS = {
            "order": [(0, 0, 0)],
            "seasonal_order": [
                (1, 0, 0, 5),
                (1, 0, 0, 7),
                (1, 0, 0, 30),
                (1, 0, 0, 31),
                (1, 0, 0, 365),
                (1, 0, 0, 91),
                (1, 0, 0, 24),
                (1, 0, 0, 60),
                (1, 0, 0, 12),
            ],
            "trend": ["n", "c", "t", "ct"],
        }

    else:
        PARAMS = {}

    return PARAMS


def estim(model):
    if model == "DecisionTreeRegressor":
        return DecisionTreeRegressor()
    elif model == "LinearRegressor":
        return LinearRegression()
    elif model == "HuberRegressor":
        return HuberRegressor()
    elif model == "TheilSenRegressor":
        return TheilSenRegressor()
    elif model == "GradientBoostingRegressor":
        return GradientBoostingRegressor()
    elif model == "PassiveAggressiveRegressor":
        return PassiveAggressiveRegressor()
    elif model == "OrthogonalMatchingPursuit":
        return OrthogonalMatchingPursuit()
    elif model == "MLPRegressor":
        return MLPRegressor()
    elif model == "RandomForestRegressor":
        return RandomForestRegressor()
    elif model == "KNeighborsRegressor":
        return KNeighborsRegressor()
    elif model == "BaggingRegressor":
        return BaggingRegressor()
    elif model == "ElasticNet":
        return ElasticNet()
    elif model == "AdaboostRegressor":
        return AdaBoostRegressor()
    elif model == "ARDRegression":
        return ARDRegression()
    else:
        raise ValueError(f"No such estimator: {model}.")
