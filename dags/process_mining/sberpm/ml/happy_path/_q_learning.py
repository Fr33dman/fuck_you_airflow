from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Optional, Union

from numpy import argmax
from numpy import max as np_max
from numpy import full as np_full
from numpy.random import uniform, choice
from pandas import DataFrame

class QLearning:
    def __init__(
        self,
        gamma: float = 0.9,
        alpha: float = 0.09,
        epsilon: float = 0.5,
        eps_scaling: float = 0.9992,
        penalty=0.0,
        short_path=None,
        time_break=10800,
    ):
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.alpha = alpha
        self.default_epsilon = epsilon
        self.epsilon = epsilon
        self.eps_scaling = eps_scaling
        self.result = None
        self.gamma = gamma
        self.penalty = penalty
        self._short_path = 1 if short_path is None else short_path
        self.counter = 0
        self._time_break = time_break

    def _get_q_value(self, state: int, action: int) -> float:
        return self.q_values[state][action]

    def _set_q_value(self, state: int, action: int, value: float) -> None:
        self.q_values[state][action] = value

    def _get_action_value(self, state: int, legal_actions) -> float:
        possible_actions = legal_actions[state]
        action_values = map(lambda action: self._get_q_value(state, action), possible_actions)

        return max(action_values) if possible_actions else 0.0

    def _update(self, state: int, action: int, reward: float, next_state: int, legal_actions) -> None:
        gamma = self.gamma
        learning_rate = self.alpha
        new_q_value = (1 - learning_rate) * self._get_q_value(state, action) + learning_rate * (
            reward + gamma * self._get_action_value(next_state, legal_actions)
        )

        self._set_q_value(state, action, new_q_value)

    def _get_best_action(self, state: int, legal_actions) -> Union[int, None]:
        possible_actions = legal_actions[state]
        q_values = map(lambda action: self._get_q_value(state, action), possible_actions)

        return possible_actions[argmax([*q_values])] if possible_actions else None

    def _get_action(self, state: int, legal_actions, state_probs) -> Union[int, None]:
        epsilon = self.epsilon
        prob_random = uniform(0, 1)
        possible_actions = legal_actions[state]

        if not possible_actions:
            return None

        return (
            choice(possible_actions, p=state_probs[state])
            if prob_random < epsilon
            else self._get_best_action(state, legal_actions)
        )

    def generate_subsession(
        self,
        iteration,
        state,
        initial_state,
        key_nodes,
        transition_probs,
        rewards,
        states_dict,
        mdp,
        legal_actions,
        reversed_dict,
        state_probs,
        step,
        regime,
        time_limit,
    ):
        states, actions = [], []
        total_reward = 0.0

        transition_probs = transition_probs[iteration]
        rewards = rewards[iteration]
        states_dict = states_dict[iteration]
        legal_actions = legal_actions[iteration]
        state_probs = state_probs[iteration]
        reversed_dict = reversed_dict[iteration]
        initial_state = f"s{str(states_dict[initial_state])}"
        key_nodes = [f"s{str(states_dict[key_nodes[0]])}"]
        mdp = mdp[iteration]
        mdp._current_state = f"s{str(state)}"

        for i in range(time_limit):
            action = self._get_action(state, legal_actions, state_probs)
            new_state, reward, done, _ = mdp.step(f"a{str(action)}")

            self._update(state, action, reward, int(new_state[1:]), legal_actions)
            states.append(reversed_dict[state])
            actions.append(reversed_dict[action])

            if not done:
                total_reward += (reward) * self.gamma ** (step + i) - self.penalty

            state = int(new_state[1:])

            if regime == "dynamic" and new_state in key_nodes:
                iteration += 1
                return states, actions, total_reward, iteration, state, done, step + i + 1
            elif (regime == "static") and (new_state in key_nodes):
                self.counter += 1

            if (regime == "static") and (new_state in key_nodes) and (self.counter >= self._short_path):
                iteration = -2
                return states, actions, total_reward, iteration, state, done, step + i + 1

            if done:
                return states + [reversed_dict[state]], actions, total_reward, iteration, state, done, step + i + 1

    def generate_session(
        self,
        initial_state,
        key_nodes,
        transition_probs,
        rewards,
        states_dict,
        mdp,
        legal_actions,
        state_probs,
        reversed_dict,
        regime,
        time_limit,
    ):
        done = False
        states_list = []
        actions_list = []
        total_reward = 0.0
        max_iter = np_max(list(states_dict))

        if regime == "static":
            state, iteration = states_dict[0][initial_state], 0
        elif regime == "dynamic":
            state, iteration = states_dict[1][initial_state], 1

        step = 0
        for _ in range(1000):
            if (regime == "dynamic") and (iteration not in states_dict):
                iteration += 1
                continue

            try:
                states, actions, reward, iteration, state, done, step = self.generate_subsession(
                    iteration=iteration,
                    state=state,
                    initial_state=initial_state,
                    key_nodes=key_nodes,
                    transition_probs=transition_probs,
                    rewards=rewards,
                    states_dict=states_dict,
                    legal_actions=legal_actions,
                    reversed_dict=reversed_dict,
                    state_probs=state_probs,
                    mdp=mdp,
                    time_limit=time_limit,
                    step=step,
                    regime=regime,
                )
            except Exception:
                continue

            states_list.extend(states)
            actions_list.extend(actions)
            total_reward += reward

            if iteration > max_iter:
                iteration -= 1

            if done:
                break

        return states_list, actions_list, total_reward, done

    def fit(
        self,
        initial_state,
        key_nodes,
        transition_probs,
        rewards,
        states_dict,
        legal_actions,
        reversed_dict,
        regime,
        state_probs,
        mdp,
        time_limit,
        n_iter: int = 100,
    ) -> DataFrame:
        now = datetime.now()
        session_rewards = []
        reconstruction_result = DataFrame(columns=["trace", "actions", "reward", "done"])

        for _ in range(n_iter):
            states_list, actions_list, session_reward, done = self.generate_session(
                initial_state=initial_state,
                key_nodes=key_nodes,
                transition_probs=transition_probs,
                rewards=rewards,
                states_dict=states_dict,
                legal_actions=legal_actions,
                state_probs=state_probs,
                reversed_dict=reversed_dict,
                mdp=mdp,
                time_limit=time_limit,
                regime=regime,
            )

            session_rewards.append(session_reward)
            reconstruction_result.loc[len(reconstruction_result)] = [
                states_list,
                actions_list,
                session_reward,
                done,
            ]
            self.epsilon *= self.eps_scaling

            if (datetime.now() - now).seconds > self._time_break:
                print(
                    f"Достигнуто максимальное время выполнения {self._time_break} секунд. Переход к следующему методу."
                )
                break

        return reconstruction_result

    def reset(self) -> None:
        self.epsilon = self.default_epsilon
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0.0))

    def _get_optimal_paths(self) -> DataFrame:
        auxiliary_df = self.result[self.result["done"] == True][self.result.reward == self.result.reward.max()]

        return auxiliary_df.loc[auxiliary_df.astype(str).drop_duplicates().index]

    def apply(
        self,
        initial_state,
        key_nodes,
        transition_probs,
        rewards,
        states_dict,
        legal_actions,
        state_probs,
        mdp,
        regime,
        reversed_dict,
        n_iter: Optional[int] = None,
        time_limit=10**4,
    ):
        if n_iter is None:
            default_n_iter = 10000
            n_iter = default_n_iter

        self.result = self.fit(
            n_iter=n_iter,
            initial_state=initial_state,
            key_nodes=key_nodes,
            transition_probs=transition_probs,
            regime=regime,
            rewards=rewards,
            states_dict=states_dict,
            legal_actions=legal_actions,
            state_probs=state_probs,
            mdp=mdp,
            time_limit=time_limit,
            reversed_dict=reversed_dict,
        )

        q_values = deepcopy(self.q_values)
        policy = np_full([np_max(list(legal_actions[-1])) + 1, np_max(list(legal_actions[-1])) + 1], 0)

        for policy_row in range(policy.shape[0]):
            for policy_column in range(policy.shape[1]):
                policy[policy_row, policy_column] = q_values[policy_row][policy_column]

        return self._get_optimal_paths(), policy
