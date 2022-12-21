from __future__ import annotations

import copy
from datetime import datetime

from numpy import where, zeros
from numpy import max as np_max
from numpy import sum as np_sum
from numpy import percentile as np_percentile
from numpy import full as np_full
from numpy.random import choice as np_choice

from sklearn.preprocessing import normalize


class CrossEntropy:
    def __init__(
        self,
        n_sessions=100,
        percentile=20,
        learning_rate=0.2,
        n_learning_events=50,
        gamma=0.9,
        penalty=0.0,
        short_path=None,
        time_break=10800,
    ):

        self.n_learning_events = n_learning_events
        self.gamma = gamma
        self.n_sessions = n_sessions
        self.percentile = percentile
        self.learning_rate = learning_rate
        self.penalty = penalty
        self._short_path = 1 if short_path is None else short_path
        self.counter = 0
        self._time_break = time_break

    @staticmethod
    def initialize_policy(legal_actions, iteration):
        policy = np_full([np_max(list(legal_actions[-1])) + 1, np_max(list(legal_actions[-1])) + 1], 0)

        for legal_action, bound_actions in legal_actions[iteration].items():
            for transition_action in bound_actions:
                policy[legal_action, transition_action] = 1

        return normalize(policy, axis=1, norm="l1")

    def generate_subsession(
        self,
        iteration,
        state,
        initial_state,
        key_nodes,
        transition_probs,
        rewards,
        states_dict,
        reversed_dict,
        legal_actions,
        mdp,
        step,
        policy,
        regime,
        time_limit=10**4,
    ):
        states, actions, states_cod, actions_cod = [], [], [], []
        total_reward = 0.0
        transition_probs = transition_probs[iteration]
        rewards = rewards[iteration]
        states_dict = states_dict[iteration]
        reversed_dict = reversed_dict[iteration]
        initial_state = f"s{str(states_dict[initial_state])}"
        key_nodes = [f"s{str(states_dict[key_nodes[0]])}"]
        mdp = mdp[iteration]
        mdp._current_state = f"s{str(state)}"
        policy = normalize(
            where(self.initialize_policy(legal_actions, iteration) > 0, policy, 0), axis=1, norm="l1"
        )

        for i in range(time_limit):
            action = np_choice(range(1, np_max(list(legal_actions[-1])) + 1), p=policy[state][1:])
            new_state, reward, done, _ = mdp.step(f"a{str(action)}")

            states.append(reversed_dict[state])
            actions.append(reversed_dict[action])
            states_cod.append(state)
            actions_cod.append(action)

            if not done:
                total_reward += (reward) * self.gamma ** (step + i) - self.penalty
            state = int(new_state[1:])

            if regime == "dynamic" and new_state in key_nodes:
                iteration += 1

                return (
                    states_cod,
                    states,
                    actions_cod,
                    actions,
                    total_reward,
                    iteration,
                    state,
                    done,
                    step + i + 1,
                )
            elif (regime == "static") and (new_state in key_nodes):
                self.counter += 1

            if (regime == "static") and (new_state in key_nodes) and (self.counter >= self._short_path):
                iteration = -2

                return (
                    states_cod,
                    states,
                    actions_cod,
                    actions,
                    total_reward,
                    iteration,
                    state,
                    done,
                    step + i + 1,
                )

            if done:
                return (
                    states_cod + [state],
                    states + [reversed_dict[state]],
                    actions_cod,
                    actions,
                    total_reward,
                    iteration,
                    state,
                    done,
                    step + i + 1,
                )

    def generate_session(
        self,
        initial_state,
        key_nodes,
        transition_probs,
        rewards,
        states_dict,
        reversed_dict,
        legal_actions,
        mdp,
        policy,
        regime,
        time_limit=10**4,
    ):
        done = False
        max_iter = np_max(list(states_dict))
        states_list, actions_list, states_cod_list, actions_cod_list = [], [], [], []
        total_reward = 0.0
        if regime == "static":
            state, iteration = states_dict[0][initial_state], 0
        elif regime == "dynamic":
            state, iteration = states_dict[1][initial_state], 1
        step = 0

        for _ in range(1000):
            if iteration not in states_dict:
                iteration += 1
                continue

            try:
                (
                    states_cod,
                    states,
                    actions_cod,
                    actions,
                    reward,
                    iteration,
                    state,
                    done,
                    step,
                ) = self.generate_subsession(
                    iteration=iteration,
                    state=state,
                    initial_state=initial_state,
                    key_nodes=key_nodes,
                    transition_probs=transition_probs,
                    rewards=rewards,
                    states_dict=states_dict,
                    mdp=mdp,
                    legal_actions=legal_actions,
                    time_limit=time_limit,
                    reversed_dict=reversed_dict,
                    step=step,
                    policy=policy,
                    regime=regime,
                )
            except Exception:
                continue

            states_list.extend(states)
            actions_list.extend(actions)
            states_cod_list.extend(states_cod)
            actions_cod_list.extend(actions_cod)
            total_reward += reward

            if iteration > max_iter:
                iteration -= 1
            if done:
                break

        return states_list, states_cod_list, actions_list, actions_cod_list, total_reward, done

    def select_elites(self, states_batch, actions_batch, rewards_batch):
        reward_threshold = np_percentile(rewards_batch, self.percentile)
        elite_states = []

        for session_states, session_reward in zip(states_batch, rewards_batch):
            if session_reward < reward_threshold:
                continue

            elite_states.extend(session_states)

        elite_actions = []
        for session_actions, session_reward in zip(actions_batch, rewards_batch):
            if session_reward < reward_threshold:
                continue

            elite_actions.extend(session_actions)

        return elite_states, elite_actions

    def get_new_policy(self, elite_states, elite_actions, initial_policy):
        new_policy = zeros([initial_policy.shape[0], initial_policy.shape[1]])
        for state, action in zip(elite_states, elite_actions):
            new_policy[state, action] += 1

        new_policy = normalize(new_policy, axis=1, norm="l1")
        for state in where(np_sum(new_policy, axis=1) < 1):
            new_policy[state, :] = initial_policy[state, :]

        return new_policy

    def apply(
        self,
        key_nodes,
        rewards,
        transition_probs,
        states_dict,
        legal_actions,
        mdp,
        reversed_dict,
        initial_state,
        regime,
    ):

        now = datetime.now()
        total_sessions = []

        policy = self.initialize_policy(legal_actions, -1)
        initial_policy = copy.deepcopy(policy)

        for _ in range(self.n_learning_events):
            sessions = [
                self.generate_session(
                    initial_state=initial_state,
                    reversed_dict=reversed_dict,
                    key_nodes=key_nodes,
                    transition_probs=transition_probs,
                    rewards=rewards,
                    states_dict=states_dict,
                    legal_actions=legal_actions,
                    mdp=mdp,
                    policy=policy,
                    regime=regime,
                    time_limit=10**4,
                )
                for _ in range(self.n_sessions)
            ]

            states_cod_batch = [session_states_cod for _, session_states_cod, _, _, _, _ in sessions]
            actions_cod_batch = [session_actions_cod for _, _, _, session_actions_cod, _, _ in sessions]
            rewards_batch = [session_reward for _, _, _, _, session_reward, _ in sessions]
            # * for session_states, session_states_cod, session_actions, session_actions_cod, session_reward, done in sessions

            total_sessions.extend(sessions)

            elite_states, elite_actions = self.select_elites(states_cod_batch, actions_cod_batch, rewards_batch)
            new_policy = self.get_new_policy(elite_states, elite_actions, initial_policy=initial_policy)
            policy = self.learning_rate * new_policy + (1 - self.learning_rate) * policy

            if (datetime.now() - now).seconds > self._time_break:
                print(
                    f"Достигнуто максимальное время выполнения {self._time_break} секунд. Переход к следующему методу."
                )
                break

        return total_sessions, policy
