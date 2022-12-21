from __future__ import annotations

from ast import literal_eval
from copy import deepcopy
from typing import Any

from loguru import logger

from numpy import NaN, argmax
from numpy import max as np_max
from numpy import maximum as max_of_arrays
from numpy import minimum as min_of_arrays
from numpy import prod, sqrt
from numpy import sum as np_sum
from pandas import DataFrame, concat

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.ml.happy_path import (
    CrossEntropy,
    genetic_algorithm,
    QLearning,
    ValueIteration,
    create_dicts,
)


def filter_data_on_column_value(
    data: DataFrame, data_filtering_column: None | str, filtering_value: Any, activity_column: str
) -> DataFrame:
    common_error_msg = (
        f"Вы выбрали 'data_filtering_column': {data_filtering_column} и 'filtering_value': {filtering_value}.\n"
        "Необходимо задать колонку, и значение внутри этой колонки для проведения фильтрации."
    )

    if data_filtering_column not in data or data_filtering_column == activity_column:
        raise ValueError(
            f"'data_filtering_column' должна быть в данных, не может быть колонкой с этапами (activity_column).\n\n"
            f"{common_error_msg}"
        )
    if filtering_value is None or filtering_value not in data[data_filtering_column]:
        raise ValueError(f"Значение 'filtering_value' не найдено в данных или пустое.\n\n{common_error_msg}")

    return data.query(f"{data_filtering_column} == '{filtering_value}'")


class HappyPath:
    def __init__(
        self,
        data_holder: DataHolder,
        key_node: str,
        initial_state: None | str = None,
        end: None | str = None,
        data_filtering_column: None | str = None,
        filtering_value: Any = None,
        reward_for_key: float = 10,
        reward_for_end: float = 0.0,
        prob_increase: float = 0.5,
        clear_outliers: float = 0.05,
        short_path: None | int = None,
        gamma: float = 1,
        mode: str = "complete",  # short, normal, long, complete
        regime: str = "static",  # dynamic or static
        penalty: float = 0.0,
        time_break=10800,
        output_algo_params=False,
        # RL гиперпараметры
        mut_list_gen: list[float] = [0.9, 0.12],
        children_list_gen: list[int] = [110, 80],
        iters_list_gen: list[int] = [20, 30],
        percentile_list_cem: list[int] = [15, 20],
        learning_list_cem: list[float] = [0.1, 0.15],
        iters_list_cem: list[int] = [80, 100],
        alpha_list: list[float] = [0.11, 0.18],
        epsilon_list_q: list[float] = [0.9, 0.12],
        iters_list_q: list[int] = [200, 300],
        iters_list_vi: list[int] = [10, 15],
        num_iter_list_vi: list[int] = [200, 300],
    ) -> None:
        self._data_holder = data_holder.copy()
        self._key_nodes = [key_node]
        self._initial_state = initial_state or "startevent"
        self._end = end or "endevent"

        if data_filtering_column is not None or filtering_value is not None:
            self._data_holder.data = filter_data_on_column_value(
                self._data_holder.data, data_filtering_column, filtering_value, self._data_holder.activity_column
            )
            logger.info(
                f"После фильтрации по 'filtering_value' осталось {self._data_holder.data.shape[0]} записей"
            )
            if key_node not in self._data_holder.data[self._data_holder.activity_column]:
                logger.error(
                    f"В данных не осталось этапов для переданной 'key_node' {key_node}.\n"
                    "Необходимо изменить 'key_node' или аргументы для фильтрации."
                )

        self._reward_for_key = reward_for_key
        self._reward_for_end = reward_for_end
        self._prob_increase = prob_increase
        self._clear_outliers = clear_outliers
        self._short_path = 1 if short_path is None else short_path
        self._mode = mode
        self._time_break = time_break
        self._output_algo_params = output_algo_params

        self._mut_list_gen = mut_list_gen
        self._children_list_gen = children_list_gen
        self._iters_list_gen = iters_list_gen
        self._percentile_list_cem = percentile_list_cem
        self._learning_list_cem = learning_list_cem
        self._alpha_list = alpha_list
        self._epsilon_list_q = epsilon_list_q
        self._iters_list_q = iters_list_q
        self._gamma = gamma
        self._iters_list_vi = iters_list_vi
        self._num_iter_list_vi = num_iter_list_vi
        self._regime = regime
        self._penalty = penalty
        self._iters_list_cem = iters_list_cem

        self._df_rl = DataFrame()
        self._best_algo = None
        self._best_path = None
        self._best_rew = None
        self._policy = None
        self._policy_alg = None
        self.dict_policy = {}
        self._states_dict = None
        self._reversed_dict = None
        self._transition_probs = None
        self._legal_actions = None
        self._mdp = None

        self.success_flag = False
        self.counter = 0

    def run_Q(
        self,
        rewards,  # Награды
        transition_probs,  # Вероятности перехода при выполнении действия
        states_dict,  # Словарь состояний
        reversed_dict,  # Обратный словарь состояний
        legal_actions,  # Приемлемые действия из каждого состояния
        state_probs,  # Вероятности перехода
        mdp,  # Марковская среда
    ):
        df_Q, policy_Q = DataFrame([]), []

        for alpha in self._alpha_list:
            for epsilon in self._epsilon_list_q:
                for q_iters in self._iters_list_q:
                    try:
                        dict_Q = {
                            "params": str(
                                dict(
                                    alpha=alpha,
                                    epsilon=epsilon,
                                    iters=q_iters,
                                )
                            )
                        }
                        q_learning = QLearning(
                            gamma=self._gamma,
                            alpha=alpha,
                            epsilon=epsilon,
                            eps_scaling=0.9992,
                            penalty=self._penalty,
                            short_path=self._short_path,
                            time_break=self._time_break,
                        )

                        path, policy = q_learning.apply(
                            n_iter=q_iters,
                            initial_state=self._initial_state,
                            key_nodes=self._key_nodes,
                            transition_probs=transition_probs,
                            regime=self._regime,
                            rewards=rewards,
                            states_dict=states_dict,
                            legal_actions=legal_actions,
                            state_probs=state_probs,
                            mdp=mdp,
                            reversed_dict=reversed_dict,
                        )
                        path_Q, rew_Q, done_Q, policy_Q = (
                            path["trace"].iloc[0],
                            path["reward"].iloc[0],
                            path["done"].iloc[0],
                            policy,
                        )
                    except Exception:
                        path_Q, rew_Q, done_Q, policy_Q = [], -1e9, False, None

                    dict_Q.update(
                        dict(
                            rew=[rew_Q],
                            path=[path_Q],
                            done=[done_Q],
                        )
                    )

                    if policy_Q is not None:
                        self.dict_policy["Q-Learning"] = policy_Q

                    df = DataFrame.from_dict(dict_Q, orient="columns")
                    df_Q = concat([df_Q, df], axis=0, sort=False)

        return df_Q

    def run_cem(
        self,
        rewards,  # Награды
        transition_probs,  # Вероятности перехода при выполнении действия
        states_dict,  # Словарь состояний
        reversed_dict,  # Обратный словарь состояний
        legal_actions,  # Приемлемые действия из каждого состояния
        mdp,  # Марковская среда
    ):
        df_cem, policy_CEM = DataFrame([]), []

        for cem_percentile in self._percentile_list_cem:
            for cem_learning in self._learning_list_cem:
                for cem_iters in self._iters_list_cem:
                    try:
                        dict_cem = {
                            "params": str(
                                dict(
                                    percentile=cem_percentile,
                                    learn_rate=cem_learning,
                                    iters=cem_iters,
                                )
                            )
                        }
                        cross_entropy = CrossEntropy(
                            n_sessions=100,
                            percentile=cem_percentile,
                            learning_rate=cem_learning,
                            n_learning_events=cem_iters,
                            gamma=self._gamma,
                            penalty=self._penalty,
                            short_path=self._short_path,
                            time_break=self._time_break,
                        )

                        sessions, policy = cross_entropy.apply(
                            key_nodes=self._key_nodes,
                            rewards=rewards,
                            transition_probs=transition_probs,
                            states_dict=states_dict,
                            initial_state=self._initial_state,
                            legal_actions=legal_actions,
                            mdp=mdp,
                            reversed_dict=reversed_dict,
                            regime=self._regime,
                        )
                        done_sessions = [session for session in sessions if session[-1]]
                        best = done_sessions[argmax([i[-2] for i in done_sessions])]

                        path_CEM, rew_CEM, done_CEM, policy_CEM = (
                            best[0],
                            best[-2],
                            best[-1],
                            policy,
                        )
                    except Exception:
                        path_CEM, rew_CEM, done_CEM, policy_CEM = [], -1e9, False, None

                    dict_cem.update(
                        dict(
                            rew=[rew_CEM],
                            done=[done_CEM],
                            path=[path_CEM],
                        )
                    )
                    if policy_CEM is not None:
                        self.dict_policy["Cross Entropy"] = policy_CEM
                    df = DataFrame.from_dict(dict_cem, orient="columns")
                    df_cem = concat([df_cem, df], axis=0, sort=False)

        return df_cem

    def run_genetic(
        self,
        rewards,  # Награды
        transition_probs,  # Вероятности перехода при выполнении действия
        states_dict,  # Словарь состояний
        reversed_dict,  # Обратный словарь состояний
        legal_actions,  # Приемлемые действия из каждого состояния
        mdp,  # Марковская среда
    ):
        df_genetic = DataFrame([])

        for mutation in self._mut_list_gen:
            for child in self._children_list_gen:
                for iters in self._iters_list_gen:
                    try:
                        dict_genetic = {
                            "params": str(
                                dict(
                                    mutation=mutation,
                                    childer=child,
                                    iters=iters,
                                )
                            )
                        }
                        path_genetic, rew_genetic, done_genetic = genetic_algorithm(
                            muta=mutation,
                            children=child,
                            iters=iters,
                            gamma=self._gamma,
                            initial_state=self._initial_state,
                            key_nodes=self._key_nodes,
                            rewards=rewards,
                            transition_probs=transition_probs,
                            states_dict=states_dict,
                            legal_actions=legal_actions,
                            mdp=mdp,
                            reversed_dict=reversed_dict,
                            regime=self._regime,
                            penalty=self._penalty,
                            end=self._end,
                            short_path=self._short_path,
                            time_break=self._time_break,
                        )
                    except Exception:
                        path_genetic, rew_genetic, done_genetic = [], -1e9, False

                    dict_genetic.update(
                        dict(
                            rew=[rew_genetic],
                            path=[path_genetic],
                            done=[done_genetic],
                        )
                    )
                    df = DataFrame.from_dict(dict_genetic, orient="columns")
                    df_genetic = concat([df_genetic, df], axis=0, sort=False)

        return df_genetic

    def run_vi(
        self,
        rewards,  # Награды
        transition_probs,  # Вероятности перехода при выполнении действия
        states_dict,  # Словарь состояний
        reversed_dict,  # Обратный словарь состояний
        mdp,  # Марковская среда
    ):
        df_vi = DataFrame([])

        for num_iter_vi in self._num_iter_list_vi:
            for iters_vi in self._iters_list_vi:
                try:
                    dict_vi = {
                        "params": str(
                            dict(
                                num_iter=num_iter_vi,
                                iters=iters_vi,
                            )
                        )
                    }
                    value_iteration = ValueIteration(
                        num_iter=num_iter_vi,
                        n_sessions=1,
                        gamma=self._gamma,
                        n_learning_events=iters_vi,
                        penalty=self._penalty,
                        short_path=self._short_path,
                        time_break=self._time_break,
                    )

                    sessions = value_iteration.apply(
                        initial_state=self._initial_state,
                        key_nodes=self._key_nodes,
                        rewards=rewards,
                        transition_probs=transition_probs,
                        reversed_dict=reversed_dict,
                        states_dict=states_dict,
                        mdp=mdp,
                        regime=self._regime,
                    )
                    done_sessions = [session for session in sessions if session[-1]]
                    best = done_sessions[argmax([i[-2] for i in done_sessions])]
                    path_vi, rew_vi, done_vi = best[0], best[-2], best[-1]
                except Exception:
                    path_vi, rew_vi, done_vi = [], -1e9, False

                dict_vi.update(
                    dict(
                        rew=[rew_vi],
                        path=[path_vi],
                        done=[done_vi],
                    )
                )
                df = DataFrame.from_dict(dict_vi, orient="columns")
                df_vi = concat([df_vi, df], axis=0, sort=False)

        return df_vi

    def _best_route(
        self,
        rewards,  # Награды
        transition_probs,  # Вероятности перехода при выполнении действия
        states_dict,  # Словарь состояний
        reversed_dict,  # Обратный словарь состояний
        legal_actions,  # Приемлемые действия из каждого состояния
        state_probs,  # Вероятности перехода
        mdp,  # Марковская среда
    ):
        modes_names_list = ["short", "normal", "long", "complete"]
        mode = modes_names_list.index(self._mode)

        logger.info("Отработка алгоритма Q-learning")
        df_Q = self.run_Q(
            rewards=rewards,  # Награды
            transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
            states_dict=states_dict,  # Словарь состояний
            reversed_dict=reversed_dict,  # Обратный словарь состояний
            legal_actions=legal_actions,  # Приемлемые действия из каждого состояния
            state_probs=state_probs,  # Вероятности перехода
            mdp=mdp,
        )
        logger.info("Отработка алгоритма Cross Entropy")
        if mode > 0:
            df_cem = self.run_cem(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                legal_actions=legal_actions,  # Приемлемые действия из каждого состояния
                mdp=mdp,
            )
        logger.info("Отработка алгоритма Genetic")
        if mode > 1:
            df_genetic = self.run_genetic(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                legal_actions=legal_actions,  # Приемлемые действия из каждого состояния
                mdp=mdp,
            )
        logger.info("Отработка алгоритма Value Iteration")
        if mode > 2:
            df_vi = self.run_vi(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                mdp=mdp,
            )

        rl_algorithms = {}
        if mode > 2:
            rl_algorithms["Value_Iteration"] = dict(
                path=df_vi[df_vi["rew"] == df_vi["rew"].max()].iloc[0]["path"],
                rew=df_vi[df_vi["rew"] == df_vi["rew"].max()].iloc[0]["rew"],
                done=df_vi[df_vi["rew"] == df_vi["rew"].max()].iloc[0]["done"],
                params=df_vi[df_vi["params"] == df_vi["params"].max()].iloc[0]["params"],
            )
        if mode > 1:
            rl_algorithms["Genetic"] = dict(
                path=df_genetic[df_genetic["rew"] == df_genetic["rew"].max()].iloc[0]["path"],
                rew=df_genetic[df_genetic["rew"] == df_genetic["rew"].max()].iloc[0]["rew"],
                done=df_genetic[df_genetic["rew"] == df_genetic["rew"].max()].iloc[0]["done"],
                params=df_genetic[df_genetic["params"] == df_genetic["params"].max()].iloc[0]["params"],
            )
        if mode > 0:
            rl_algorithms["Cross Entropy"] = dict(
                path=df_cem[df_cem["rew"] == df_cem["rew"].max()].iloc[0]["path"],
                rew=df_cem[df_cem["rew"] == df_cem["rew"].max()].iloc[0]["rew"],
                done=df_cem[df_cem["rew"] == df_cem["rew"].max()].iloc[0]["done"],
                params=df_cem[df_cem["params"] == df_cem["params"].max()].iloc[0]["params"],
            )

        rl_algorithms["Q-Learning"] = dict(
            path=df_Q[df_Q["rew"] == df_Q["rew"].max()].iloc[0]["path"],
            rew=df_Q[df_Q["rew"] == df_Q["rew"].max()].iloc[0]["rew"],
            done=df_Q[df_Q["rew"] == df_Q["rew"].max()].iloc[0]["done"],
            params=df_Q[df_Q["params"] == df_Q["params"].max()].iloc[0]["params"],
        )
        df_rl = DataFrame(rl_algorithms).T.sort_values(by="rew", ascending=False)

        best = df_rl[df_rl["rew"] == df_rl["rew"].max()].iloc[0]
        dict_policy = self.dict_policy
        best_algo = best.name
        params_dict = literal_eval(best["params"])

        logger.info("Отработка финального алгоритма")
        if best_algo == "Cross Entropy":
            self._iters_list_cem = [
                np_max(self._iters_list_cem)
                * int(
                    sqrt(len(self._iters_list_cem) * len(self._learning_list_cem) * len(self._percentile_list_cem))
                )
            ]
            self._percentile_list_cem = [params_dict["percentile"]]
            self._learning_list_cem = [params_dict["learn_rate"]]
            df_cem = self.run_cem(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                legal_actions=legal_actions,  # Приемлемые действия из каждого состояния
                mdp=mdp,
            )

            rl_cross_entropy_algorithm = dict(
                path=df_cem[df_cem["rew"] == df_cem["rew"].max()].iloc[0]["path"],
                rew=df_cem[df_cem["rew"] == df_cem["rew"].max()].iloc[0]["rew"],
                done=df_cem[df_cem["rew"] == df_cem["rew"].max()].iloc[0]["done"],
                params=str(
                    {
                        key: value
                        for key, value in literal_eval(
                            df_cem[df_cem["params"] == df_cem["params"].max()].iloc[0]["params"]
                        ).items()
                        if key != "iters"
                    }
                ),
            )
            self._df_rl = DataFrame({"Cross Entropy": rl_cross_entropy_algorithm}).T.iloc[0]

            if self._df_rl["rew"] > best["rew"]:
                policy = self.dict_policy["Q-Learning"]
            else:
                policy = dict_policy["Cross Entropy"]
                self._df_rl = best

            policy_alg = "Cross Entropy"

            return (
                self._df_rl,
                self._df_rl.name,
                self._df_rl["path"],
                self._df_rl["rew"],
                policy,
                policy_alg,
            )
        elif best_algo == "Genetic":
            self._iters_list_gen = [
                np_max(self._iters_list_gen)
                * int(sqrt(len(self._iters_list_gen) * len(self._children_list_gen) * len(self._mut_list_gen)))
            ]
            self._mut_list_gen = [params_dict["mutation"]]
            self._children_list_gen = [params_dict["children"]]
            df_genetic = self.run_genetic(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                legal_actions=legal_actions,  # Приемлемые действия из каждого состояния
                mdp=mdp,
            )

            rl_genetic_algorithm = dict(
                path=df_genetic[df_genetic["rew"] == df_genetic["rew"].max()].iloc[0]["path"],
                rew=df_genetic[df_genetic["rew"] == df_genetic["rew"].max()].iloc[0]["rew"],
                done=df_genetic[df_genetic["rew"] == df_genetic["rew"].max()].iloc[0]["done"],
                params=str(
                    {
                        k: v
                        for k, v in literal_eval(
                            df_genetic[df_genetic["params"] == df_genetic["params"].max()].iloc[0]["params"]
                        ).items()
                        if k != "iters"
                    }
                ),
            )
            self._df_rl = DataFrame({"Genetic": rl_genetic_algorithm}).T.iloc[0]

            if self._df_rl["rew"] < best["rew"]:
                self._df_rl = best

            for i, _ in df_rl.iterrows():
                if i in ["Q-Learning", "Cross Entropy"]:
                    policy = dict_policy[i]
                    policy_alg = i
                    break

            return (
                self._df_rl,
                self._df_rl.name,
                self._df_rl["path"],
                self._df_rl["rew"],
                policy,
                policy_alg,
            )
        elif best_algo == "Q-Learning":
            self._iters_list_q = [
                np_max(self._iters_list_q)
                * int(sqrt(len(self._iters_list_q) * len(self._epsilon_list_q) * len(self._alpha_list)))
            ]
            self._alpha_list = [params_dict["alpha"]]
            self._epsilon_list_q = [params_dict["epsilon"]]
            df_Q = self.run_Q(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                legal_actions=legal_actions,  # Приемлемые действия из каждого состояния
                state_probs=state_probs,  # Вероятности перехода
                mdp=mdp,
            )

            rl_q_learning_algorithm = dict(
                path=df_Q[df_Q["rew"] == df_Q["rew"].max()].iloc[0]["path"],
                rew=df_Q[df_Q["rew"] == df_Q["rew"].max()].iloc[0]["rew"],
                done=df_Q[df_Q["rew"] == df_Q["rew"].max()].iloc[0]["done"],
                params=str(
                    {
                        k: v
                        for k, v in literal_eval(
                            df_Q[df_Q["params"] == df_Q["params"].max()].iloc[0]["params"]
                        ).items()
                        if k != "iters"
                    }
                ),
            )
            self._df_rl = DataFrame({"Q-Learning": rl_q_learning_algorithm}).T.iloc[0]

            if self._df_rl["rew"] > best["rew"]:
                policy = self.dict_policy["Q-Learning"]
            else:
                policy = dict_policy["Q-Learning"]
                self._df_rl = best

            policy_alg = "Q-Learning"

            return (
                self._df_rl,
                self._df_rl.name,
                self._df_rl["path"],
                self._df_rl["rew"],
                policy,
                policy_alg,
            )
        elif best_algo == "Value_Iteration":
            self._iters_list_vi = [
                np_max(self._iters_list_vi) * int(sqrt(len(self._iters_list_vi) * len(self._num_iter_list_vi)))
            ]
            self._num_iter_list_vi = [params_dict["num_iter"]]
            df_vi = self.run_vi(
                rewards=rewards,  # Награды
                transition_probs=transition_probs,  # Вероятности перехода при выполнении действия
                states_dict=states_dict,  # Словарь состояний
                reversed_dict=reversed_dict,  # Обратный словарь состояний
                mdp=mdp,
            )

            rl_value_iteration_algorithm = dict(
                path=df_vi[df_vi["rew"] == df_vi["rew"].max()].iloc[0]["path"],
                rew=df_vi[df_vi["rew"] == df_vi["rew"].max()].iloc[0]["rew"],
                done=df_vi[df_vi["rew"] == df_vi["rew"].max()].iloc[0]["done"],
                params=str(
                    {
                        k: v
                        for k, v in literal_eval(
                            df_vi[df_vi["params"] == df_vi["params"].max()].iloc[0]["params"]
                        ).items()
                        if k != "iters"
                    }
                ),
            )
            self._df_rl = DataFrame({"Value_Iteration": rl_value_iteration_algorithm}).T.iloc[0]

            if self._df_rl["rew"] < best["rew"]:
                self._df_rl = best

            for i, _ in df_rl.iterrows():
                if i in ["Q-Learning", "Cross Entropy"]:
                    policy = dict_policy[i]
                    policy_alg = i
                    break

            return (
                self._df_rl,
                self._df_rl.name,
                self._df_rl["path"],
                self._df_rl["rew"],
                policy,
                policy_alg,
            )

    def apply(self):
        (
            rewards,  # Награды
            transition_probs,  # Вероятности перехода при выполнении действия
            states_dict,  # Словарь состояний
            reversed_dict,  # Обратный словарь состояний
            legal_actions,  # Приемлемые действия из каждого состояния
            state_probs,  # Вероятности перехода
            mdp,  # Марковская среда
        ) = create_dicts(
            data_holder=self._data_holder,
            key_nodes=self._key_nodes,
            initial_state=self._initial_state,
            reward_for_key=self._reward_for_key,
            end=self._end,
            reward_for_end=self._reward_for_end,
            prob_increase=self._prob_increase,
            clear_outliers=self._clear_outliers,
            short_path=self._short_path,
        )
        self._states_dict = states_dict
        self._reversed_dict = reversed_dict
        self._transition_probs = transition_probs
        self._legal_actions = legal_actions
        self._mdp = mdp

        (
            self._df_rl,
            self._best_algo,
            self._best_path,
            self._best_rew,
            self._policy,
            self._policy_alg,
        ) = self._best_route(
            rewards,
            transition_probs,
            states_dict,
            reversed_dict,
            legal_actions,
            state_probs,
            mdp,
        )

    def get_df_rl(self) -> DataFrame:
        print(self._policy_alg)

        if self._df_rl.empty:
            raise RuntimeError("Call apply() method for HappyPath object first.")

        return self._df_rl.copy() if self._output_algo_params else self._df_rl[self._df_rl.index != "params"]

    def get_best_algo(self) -> str:
        if self._best_algo is not None:
            return self._best_algo
        else:
            raise RuntimeError("Call apply() method for HappyPath object first.")

    def get_best_path(self) -> list[str]:
        if self._best_path is not None:
            return self._best_path
        else:
            raise RuntimeError("Call apply() method for HappyPath object first.")

    def get_best_path_df(self) -> DataFrame:
        trace_data = DataFrame(self.get_best_path(), columns=["activity"])
        current_activities, following_activities = trace_data["activity"], trace_data["activity"].shift(-1)

        return (
            trace_data.assign(
                transition=tuple(
                    zip(
                        current_activities,
                        following_activities,
                    )
                )
            )
            .groupby(by="transition", sort=False)
            .count()
            .rename(columns={"activity": "count"})
        )

    def get_best_rew(self) -> float:
        if self._best_rew is not None:
            return self._best_rew
        else:
            raise RuntimeError("Call apply() method for HappyPath object first.")

    def subsession_particular_path(self, path, iteration, step):
        probabilities_list = []
        transition_probs = self._transition_probs[iteration]
        states_dict = self._states_dict[iteration]

        for path_idx in range(step, len(path) - 1):
            current_state = states_dict[path[path_idx]]
            action = states_dict[path[path_idx + 1]]
            probability = transition_probs[f"s{str(current_state)}"][f"a{str(action)}"][f"s{str(action)}"]

            probabilities_list.append(probability)
            if self._regime == "dynamic" and path[path_idx + 1] in self._key_nodes:
                iteration += 1
                return probabilities_list, iteration, path_idx + 1

            if path_idx == len(path) - 2:
                return probabilities_list, iteration, path_idx + 1

    def probability_particular_path(self, path, time_limit=10000):

        max_iter = np_max(list(self._states_dict))
        probabilities_list = []

        if self._regime == "static":
            iteration = 0
        elif self._regime == "dynamic":
            iteration = 1

        step = 0
        for _ in range(time_limit):
            if iteration not in self._states_dict:
                iteration += 1
                continue

            try:
                probs, iteration, step = self.subsession_particular_path(path=path, iteration=iteration, step=step)
            except Exception:
                continue

            probabilities_list.extend(probs)

            if iteration > max_iter:
                iteration -= 1
            if step >= len(path) - 2:
                break

        return prod(probabilities_list)

    def success_subsession_prob(self, states_list, iteration, state, step, time_limit):
        states, actions = [], []
        mdp = self._mdp[iteration]
        mdp._current_state = f"s{str(state)}"
        sl = deepcopy(states_list)

        for i in range(time_limit):
            sl.append(self._reversed_dict[iteration][state])
            states.append(self._reversed_dict[iteration][state])

            action = self._states_dict[iteration][self.Advisor(sl)]
            # new_state, reward, done
            new_state, _, done, _ = mdp.step(f"a{str(action)}")

            if new_state in f"s{str(self._states_dict[iteration][self._key_nodes[0]])}":
                self.success_flag = True

            actions.append(self._reversed_dict[iteration][action])
            state = int(new_state[1:])

            if (
                self._regime == "dynamic"
                and new_state in f"s{str(self._states_dict[iteration][self._key_nodes[0]])}"
            ):
                iteration += 1
                return states, actions, iteration, state, done, step + i + 1
            elif (self._regime == "static") and (
                new_state in f"s{str(self._states_dict[iteration][self._key_nodes[0]])}"
            ):
                self.counter += 1

            if (
                (self._regime == "static")
                and (new_state in f"s{str(self._states_dict[iteration][self._key_nodes[0]])}")
                and (self.counter >= self._short_path)
            ):
                iteration = -2
                return states, actions, iteration, state, done, step + i + 1

            if done:
                return (
                    states + [self._reversed_dict[iteration][state]],
                    actions,
                    iteration,
                    state,
                    done,
                    step + i + 1,
                )

    def success_session_prob(self, prev_path=None):
        self.success_flag = False
        done = False
        max_iter = np_max(list(self._states_dict))
        states_list = []
        actions_list = []

        if prev_path is None:
            if self._regime == "dynamic":
                state, iteration = self._states_dict[1][self._initial_state], 1
            elif self._regime == "static":
                state, iteration = self._states_dict[0][self._initial_state], 0
        elif self._regime in ["static", "dynamic"]:
            state, iteration = self._states_dict[0][prev_path[-1]], 0

        step = 0
        for _ in range(10000):
            if (self._regime == "dynamic") and (iteration not in self._states_dict):
                iteration += 1
                continue

            (states, actions, iteration, state, done, step,) = self.success_subsession_prob(
                states_list=states_list,
                iteration=iteration,
                state=state,
                time_limit=10000,
                step=step,
            )

            states_list.extend(states)
            actions_list.extend(actions)

            if iteration > max_iter:
                iteration -= 1
            if done:
                break

        return states_list, actions_list, done, self.success_flag

    def Advisor(self, prev_path):
        key_node = self._key_nodes[0]
        state = prev_path[-1]
        best_actions = {}

        for k in self._legal_actions:
            best_actions[k] = {}
            for i in self._legal_actions[k]:
                try:
                    best_actions[k][i] = self._legal_actions[k][i][
                        argmax([self._policy[i][j] for j in self._legal_actions[k][i]])
                    ]
                except ValueError:
                    best_actions[k][i] = NaN

        iteration = prev_path.count(key_node)
        max_iter = np_max(list(self._states_dict))
        next_iteration = min_of_arrays(max_iter, iteration + 1)

        if state == "endevent":
            return "END of the route"
        if prev_path == self._best_path[: len(prev_path)]:
            return self._best_path[len(prev_path)]

        list_iter_in_best = [i for i, x in enumerate(self._best_path) if x == key_node]
        ind_n_iter_in_best = (
            list_iter_in_best[min_of_arrays(iteration - 1, len(list_iter_in_best) - 1)] if iteration > 0 else 0
        )

        ind_iter_in_prev = (
            [i for i, x in enumerate(prev_path) if x == key_node][iteration - 1] if iteration > 0 else 0
        )

        if iteration <= len(list_iter_in_best) - 1:
            ind_nplus1_iter_in_best = list_iter_in_best[iteration]
        else:
            ind_nplus1_iter_in_best = -1

        future_path = self._best_path[ind_n_iter_in_best : ind_nplus1_iter_in_best + 1]
        if state not in future_path:
            return (
                self._reversed_dict[next_iteration][
                    best_actions[next_iteration][self._states_dict[next_iteration][state]]
                ]
                if self._regime == "dynamic"
                else self._reversed_dict[0][best_actions[0][self._states_dict[0][state]]]
            )

        list_state_future_path = [i for i, x in enumerate(future_path) if x == state]
        ind_state_prev_path = max_of_arrays(
            len([i for i, x in enumerate(prev_path[ind_iter_in_prev + 1 :]) if x == state]) - 1,
            0,
        )
        try:
            ind_state_prev_path = min_of_arrays(len(list_state_future_path) - 1, ind_state_prev_path)
            act = future_path[list_state_future_path[ind_state_prev_path] + 1]

        except IndexError:
            act = future_path[list_state_future_path[-1] + 1]
        return act

    def prob_of_success(self, prev_path, n_iters=3000):
        sessions = [self.success_session_prob(prev_path) for _ in range(n_iters)]

        return np_sum([item[-1] for item in sessions]) / len(sessions)
