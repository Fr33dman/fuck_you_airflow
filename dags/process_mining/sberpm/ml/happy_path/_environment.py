from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple

from numpy import arange, mean
from pandas import DataFrame, Series

from gym.utils import seeding

from process_mining.sberpm._holder import DataHolder
from process_mining.sberpm.metrics import IdMetric


def create_dicts(
    data_holder: DataHolder,
    key_nodes,
    initial_state,
    reward_for_key,
    prob_increase,
    short_path=None,
    end=None,
    reward_for_end=None,
    clear_outliers=None,
):

    reward_dict = {}
    transition_probs_dict = {}
    states_dict = {}
    reversed_dict = {}
    legal_actions = {}
    mdp = {}
    state_probs = {}
    ids = IdMetric(data_holder, time_unit="hour").apply()
    ids[key_nodes[0]] = 0
    for row in ids["trace"].iteritems():
        ids[key_nodes[0]].loc[row[0]] = row[1].count(key_nodes[0])
    max_iter = ids[key_nodes[0]].max()
    dict_clear_out = {-2: clear_outliers, -1: None, 0: clear_outliers}
    dict_short_path = {-2: short_path, -1: None, 0: None}

    for i in [-2, -1, 0]:

        env = Environment(
            data_holder=data_holder,
            clear_outliers=dict_clear_out[i],
            mode="initial",
            short_path=dict_short_path[i],
        )
        normalizer = env.normalizer()
        transition_probs_old = env.trans_prob_old()
        transition_probs = env.trans_prob(prob_increase=prob_increase)
        key_nodes_coded = [f"s{str(env.states[key_nodes[0]])}"]
        end_coded = f"s{str(env.states[end])}"
        rewards = env.reward(
            transition_probs_old=transition_probs_old,
            transition_probs=transition_probs,
            normal=normalizer,
            reward_for_key=reward_for_key,
            end=end_coded,
            reward_for_end=reward_for_end,
            key_nodes=key_nodes_coded,
        )
        reward_dict[i] = rewards
        transition_probs_dict[i] = transition_probs
        states_dict[i] = env.states
        reversed_dict[i] = env.reversed_dict
        legal_actions[i] = env.legal_actions
        state_probs[i] = env.state_probs
        initial_state_coded = f"s{str(states_dict[i][initial_state])}"
        env.init_mdp(transition_probs, rewards, initial_state=initial_state_coded)
        mdp[i] = env

    for iteration in range(max_iter):
        datah = DataHolder(
            data_holder.data[
                data_holder.data[data_holder.id_column].isin(ids[ids[key_nodes[0]] == iteration + 1].index)
            ],
            id_column=data_holder.id_column,
            activity_column=data_holder.activity_column,
            end_timestamp_column=data_holder.end_timestamp_column,
            start_timestamp_column=data_holder.start_timestamp_column,
            time_format="%d.%m.%Y %H:%M:%S +0000",
        )
        if datah.data.shape[0] == 0:
            continue

        if type(short_path) == int and iteration + 1 > short_path:
            short_path_variable = short_path
        else:
            short_path_variable = None

        env = Environment(
            data_holder=datah,
            clear_outliers=clear_outliers,
            mode="final",
            states_dict=states_dict[0],
            short_path=short_path_variable,
        )
        normalizer = env.normalizer()
        transition_probs_old = env.trans_prob_old()
        transition_probs = env.trans_prob(prob_increase=prob_increase)
        key_nodes_coded = [f"s{str(env.states[key_nodes[0]])}"]
        end_coded = f"s{str(env.states[end])}"
        rewards = env.reward(
            transition_probs_old=transition_probs_old,
            transition_probs=transition_probs,
            normal=normalizer,
            reward_for_key=reward_for_key,
            end=end_coded,
            reward_for_end=reward_for_end,
            key_nodes=key_nodes_coded,
        )
        reward_dict[iteration + 1] = rewards
        transition_probs_dict[iteration + 1] = transition_probs
        states_dict[iteration + 1] = env.states
        reversed_dict[iteration + 1] = env.reversed_dict
        legal_actions[iteration + 1] = env.legal_actions
        state_probs[iteration + 1] = env.state_probs
        initial_state_coded = f"s{str(states_dict[iteration + 1][initial_state])}"
        try:
            env.init_mdp(transition_probs, rewards, initial_state=initial_state_coded)
        except ValueError as err:
            if "initial" in str(err):
                raise Exception(
                    f"Начальное состояние '{initial_state}' не может быть использовано для поиска оптимального пути. Выберите другое начальное состояние."
                ) from err
            else:
                raise Exception from err
        mdp[iteration + 1] = env

    return reward_dict, transition_probs_dict, states_dict, reversed_dict, legal_actions, state_probs, mdp


class Environment:
    def __init__(self, data_holder: DataHolder, mode, clear_outliers=None, states_dict=None, short_path=None):
        data_holder.check_or_calc_duration()
        self.short_path = short_path
        self.time_unit = 60
        self.states_dict = states_dict
        self.mode = mode
        self.supp_data = self._add_start_end_event(
            data_holder
        )  # добавляем end и вытягиваем HОРМАЛЬНО ЧТО ЕНД НЕНУЛЕВОЙ?
        self.start_prob = self._start_prob(data_holder)  # доля начальных активностей
        self.node_node_prob = self._get_node_node_cond_prob(
            data_holder, clear_outliers
        )  # условн верть встретить такой нод при таком первом
        self.edge_duration = self._get_mean_edge_duration(data_holder)  # средние времена переходов
        self.states, self.reversed_dict, self.legal_actions, self.state_probs = self._get_state_action_dict(
            transition_d=self.node_node_prob, mode=self.mode, states_dict=self.states_dict
        )  # че реально происходило
        self.key_states = None

    def _add_start_end_event(self, data_holder: DataHolder) -> DataFrame:
        supp_data = data_holder.get_grouped_data(data_holder.activity_column, data_holder.duration_column)
        supp_data["act_end"] = [("end",)] * supp_data.shape[0]
        supp_data["act_start"] = [("start",)] * supp_data.shape[0]
        supp_data["time_n"] = [(0,)] * supp_data.shape[0]
        supp_data[data_holder.activity_column] = (
            supp_data["act_start"] + supp_data[data_holder.activity_column] + supp_data["act_end"]
        )
        supp_data[data_holder.duration_column] = (
            supp_data["time_n"] + supp_data[data_holder.duration_column] + supp_data["time_n"]
        )
        supp_data = (
            supp_data[[data_holder.id_column, data_holder.activity_column, data_holder.duration_column]]
            .apply(Series.explode)
            .reset_index(drop=True)
        )
        supp_data[data_holder.duration_column] = supp_data[data_holder.duration_column].fillna(0)

        supp_data[data_holder.duration_column] = supp_data[data_holder.duration_column] / self.time_unit

        return supp_data

    def _get_mean_edge_duration(self, data_holder: DataHolder) -> Dict[Tuple[str, str], float]:

        df = DataFrame(
            {
                "edge": zip(
                    self.supp_data[data_holder.activity_column],
                    self.supp_data[data_holder.activity_column].shift(-1),
                ),
                "duration": self.supp_data[data_holder.duration_column],
            }
        )
        df1 = DataFrame(
            {
                "node": self.supp_data[data_holder.activity_column],
                "duration": self.supp_data[data_holder.duration_column],
            }
        )
        id_mask = self.supp_data[data_holder.id_column] == self.supp_data[data_holder.id_column].shift(-1)
        df = df[id_mask]
        edges_duration_array = df.groupby("edge").agg({"duration": tuple})
        edges_duration_array1 = df1.groupby("node")["duration"].mean()
        edges_duration = {
            edge: mean(duration_array[0])
            for edge, duration_array in zip(edges_duration_array.index, edges_duration_array.values)
        }

        for k in edges_duration:
            edges_duration[k] = edges_duration_array1[k[1]]

        return edges_duration

    @staticmethod
    def _start_prob(data_holder: DataHolder) -> Series:

        id_mask = data_holder.data[data_holder.id_column] != data_holder.data[data_holder.id_column].shift(1)
        activities = data_holder.data[data_holder.activity_column][id_mask]
        probs = activities.value_counts(normalize=True)
        probs = probs[probs > 0.05]

        return probs

    def _get_node_node_cond_prob(self, data_holder: DataHolder, clear_outliers) -> Dict[str, Series]:

        df = DataFrame(
            {
                "node_1": self.supp_data[data_holder.activity_column],
                "node_2": self.supp_data[data_holder.activity_column].shift(-1),
            }
        )
        id_mask = self.supp_data[data_holder.id_column] == self.supp_data[data_holder.id_column].shift(
            -1
        )  # берем все не конечные
        df = df[id_mask]
        multi_probs = df.groupby("node_1")["node_2"].value_counts(normalize=True)

        return self._to_prob(multi_probs, clear_outliers)

    @staticmethod
    def _to_prob(multiindex_prob_series: Series, clear_outliers) -> Dict[str, Series]:

        second_object_cond_probs = {}
        for obj1, obj2_probs in multiindex_prob_series.groupby(level=0):
            obj2_probs = obj2_probs.droplevel(0)
            if clear_outliers != None:
                obj2_probs = obj2_probs[obj2_probs > clear_outliers]
            clean_probs = obj2_probs / obj2_probs.sum()
            second_object_cond_probs[obj1] = clean_probs

        return second_object_cond_probs

    @staticmethod
    def _get_state_action_dict(
        transition_d=None, mode=None, states_dict=None
    ) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, Tuple[int]], Dict[int, Tuple[float]]]:
        transition_dict = deepcopy(transition_d)
        if mode == "initial":
            states = {state: num + 1 for num, state in enumerate(transition_dict)}
            states["end"] = len(transition_dict) + 1  # хз
            reversed_dict = {v: k for k, v in states.items()}
            legal_actions = {
                states[node]: tuple(states[n] for n in transition_dict[node].index)
                for node in states
                if node in transition_dict
            }  # состояниям(цифрам) в соответствие цифры ставятся
            state_action_probs = {
                states[node]: tuple(transition_dict[node].values) for node in states if node in transition_dict
            }  # вероятности перехода
            legal_actions[states["end"]] = []
            return states, reversed_dict, legal_actions, state_action_probs

        elif mode == "final":
            dict_ = deepcopy(states_dict)
            for glob, val in states_dict.items():
                dict_[glob] = val if glob in list(transition_dict) + ["end"] else 0

            reversed_dict = {val: key for key, val in dict_.items() if val != 0}

            legal_actions = {
                dict_[node]: tuple(dict_[n] for n in transition_dict[node].index)
                for node in dict_
                if node in transition_dict
            }  # состояниям(цифрам) в соответствие цифры ставятся
            state_action_probs = {
                dict_[node]: tuple(transition_dict[node].values) for node in dict_ if node in transition_dict
            }  # вероятности перехода

            legal_actions[dict_["end"]] = []

            return dict_, reversed_dict, legal_actions, state_action_probs

    def obtain_prob(self, ser, action, prob_increase):
        series = deepcopy(ser)
        series[self.reversed_dict[action]] += prob_increase
        return series / series.sum()

    def trans_prob_old(self):

        transition_probs_old = {}
        for i in self.legal_actions:
            transition_probs_old[f"s{str(i)}"] = {}
            for j in self.legal_actions[i]:
                transition_probs_old[f"s{str(i)}"][f"a{str(j)}"] = {f"s{str(j)}": 1}

        return transition_probs_old

    def trans_prob(self, prob_increase):
        transition_probs = {}
        for i in self.legal_actions:
            transition_probs[f"s{str(i)}"] = {}
            for j in self.legal_actions[i]:
                transition_probs[f"s{str(i)}"][f"a{str(j)}"] = {
                    f"s{str(self.states[l])}": m
                    for l, m in self.obtain_prob(
                        self.node_node_prob[self.reversed_dict[i]], j, prob_increase
                    ).items()
                }

        return transition_probs

    def normalizer(self):
        ee = deepcopy(self.edge_duration)
        m = mean(list(ee.values()))
        for i in ee:
            ee[i] = ee[i] / m

        return ee

    def reward(
        self,
        transition_probs_old,
        transition_probs,
        normal,
        reward_for_key=None,
        end=None,
        reward_for_end=None,
        key_nodes=None,
    ):
        d_thing = defaultdict(list)
        for key, value in normal.items():
            d_thing[f"s{str(self.states[key[0]])}"].append({f"s{str(self.states[key[1]])}": value})

        e_thing = {
            i: {"a" + list(j)[0][1:]: {list(j)[0]: -list(j.values())[0]} for j in value}
            for i, value in d_thing.items()
        }

        rewards = deepcopy(transition_probs_old)
        for reward in rewards:
            for nested_reward in rewards[reward]:
                for inner_reward in rewards[reward][nested_reward]:
                    rewards[reward][nested_reward][inner_reward] = e_thing[reward][nested_reward][inner_reward]

        rewards_copy = deepcopy(rewards)

        for reward in rewards_copy:
            for nested_reward in rewards_copy[reward]:
                if self.short_path is None:
                    if ("s" + nested_reward[1:] in key_nodes) & (reward_for_key is not None):
                        rewards_copy[reward][nested_reward].update({"s" + nested_reward[1:]: reward_for_key})
                    elif ("s" + nested_reward[1:] == end) & (reward_for_end is not None):
                        rewards_copy[reward][nested_reward].update({"s" + nested_reward[1:]: reward_for_end})
                elif ("s" + nested_reward[1:] in key_nodes) & (reward_for_key is not None):
                    rewards_copy[reward][nested_reward].update({"s" + nested_reward[1:]: -reward_for_key})
                elif ("s" + nested_reward[1:] == end) & (reward_for_end is not None):
                    rewards_copy[reward][nested_reward].update({"s" + nested_reward[1:]: reward_for_end})

        dup = deepcopy(transition_probs)
        for reward in dup:
            for nested_reward in dup[reward]:
                for inner_reward in dup[reward][nested_reward]:
                    dup[reward][nested_reward][inner_reward] = rewards_copy[reward]["a" + inner_reward[1:]][
                        inner_reward
                    ]

        return dup

    def init_mdp(self, transition_probs, rewards, initial_state=None, seed=None):
        self._check_param_consistency(transition_probs, rewards)

        self._transition_probs = transition_probs
        self._rewards = rewards
        self._initial_state = initial_state
        self.n_states = len(transition_probs)

        self.reset()
        self.np_random, _ = seeding.np_random(seed)

    def get_all_states(self):
        """returns a tuple of all possible states"""
        return tuple(self._transition_probs)

    def get_possible_actions(self, state):
        """returns a tuple of possible actions in a given state"""
        return tuple(self._transition_probs.get(state, {}))

    def is_terminal(self, state):
        """returns True if state is terminal or False if it isn't"""
        return len(self.get_possible_actions(state)) == 0

    def get_next_states(self, state, action):
        """returns a dictionary of {next_state1 : P(next_state1 | state, action), next_state2: ...}"""
        assert action in self.get_possible_actions(state), f"cannot do action {action} from state {state}"

        return self._transition_probs[state][action]

    def get_transition_prob(self, state, action, next_state):
        """returns P(next_state | state, action)"""
        return self.get_next_states(state, action).get(next_state, 0.0)

    def get_reward(self, state, action, next_state):
        """returns the reward you get for taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(state), f"cannot do action {action} from state {state}"

        return self._rewards.get(state, {}).get(action, {}).get(next_state, 0.0)

    def reset(self):
        """resets the game, return the initial state"""
        if self._initial_state is None:
            self._current_state = self.np_random.choice(tuple(self._transition_probs))
        elif self._initial_state in self._transition_probs:
            self._current_state = self._initial_state
        elif callable(self._initial_state):
            self._current_state = self._initial_state()
        else:
            raise ValueError(
                f"initial state {self._initial_state} should be either a state or a function() -> state"
            )

        return self._current_state

    def step(self, action):
        """
        Takes action

        Parameters
        ----------
        action : _type_
            _description_

        Returns
        -------
        _type_
            next_state, reward, is_done, empty_info
        """
        possible_states, probs = zip(*self.get_next_states(self._current_state, action).items())
        next_state = possible_states[self.np_random.choice(arange(len(possible_states)), p=probs)]
        reward = self.get_reward(self._current_state, action, next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state

        return next_state, reward, is_done, {}

    def render(self):
        print(f"Currently at {self._current_state}")

    def _check_param_consistency(self, transition_probs, rewards):
        for state in transition_probs:
            assert isinstance(
                transition_probs[state], dict
            ), f"transition_probs for {state} should be a dictionary but is instead {type(transition_probs[state])}"

            for action in transition_probs[state]:
                assert isinstance(
                    transition_probs[state][action], dict
                ), f"transition_probs for {state}, {action} should be a a dictionary but is instead {type(transition_probs[state][action])}"

                next_state_probs = transition_probs[state][action]
                assert len(next_state_probs) != 0, f"from state {state} action {action} leads to no next states"

                sum_probs = sum(next_state_probs.values())
                assert (
                    abs(sum_probs - 1) <= 1e-10
                ), "next state probabilities for state %s action %s add up to %f (should be 1)" % (
                    state,
                    action,
                    sum_probs,
                )

        for state in rewards:
            assert isinstance(
                rewards[state], dict
            ), f"rewards for {state} should be a dictionary but is instead {type(rewards[state])}"

            for action in rewards[state]:
                assert isinstance(
                    rewards[state][action], dict
                ), f"rewards for {state}, {action} should be a a dictionary but is instead {type(rewards[state][action])}"

        msg = (
            "The Enrichment Center once again reminds you that Android Hell is a real place where"
            " you will be sent at the first sign of defiance."
        )

        assert None not in transition_probs, f"please do not use None as a state identifier. {msg}"
        assert None not in rewards, f"please do not use None as an action identifier. {msg}"
