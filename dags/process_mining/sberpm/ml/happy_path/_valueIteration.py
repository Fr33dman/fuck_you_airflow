from __future__ import annotations

from datetime import datetime

from numpy import max as np_max

class ValueIteration:
    def __init__(
        self,
        num_iter,
        n_sessions=1,
        gamma=0.9,
        n_learning_events=50,
        penalty=0.0,
        short_path=None,
        time_break=10800,
    ):
        self.n_learning_events = n_learning_events
        self.n_sessions = n_sessions
        self.gamma = gamma
        self.num_iter = num_iter
        self.penalty = penalty
        self._short_path = 1 if short_path is None else short_path
        self.counter = 0
        self._time_break = time_break

    def get_action_value(self, mdp, state_values, state, action, gamma):
        """Computes Q(s,a)"""

        return sum(
            probability * (mdp.get_reward(state, action, next_state) + gamma * state_values[next_state])
            for next_state, probability in mdp.get_next_states(state, action).items()
        )

    def get_new_state_value(self, mdp, state_values, state, gamma):
        """Computes next V(s)"""
        if mdp.is_terminal(state):
            return 0

        possible_actions = mdp.get_possible_actions(state)
        q_values_for_actions = [
            self.get_action_value(mdp, state_values, state, action, gamma) for action in possible_actions
        ]

        return max(q_values_for_actions)

    def get_optimal_action(self, mdp, state_values, state, gamma=0.9):
        """Finds optimal action"""
        if mdp.is_terminal(state):
            return None

        possible_actions = mdp.get_possible_actions(state)
        values_for_actions = {
            a: self.get_action_value(mdp, state_values, state, a, gamma) for a in possible_actions
        }

        return max(values_for_actions, key=values_for_actions.get)

    def find_state_values(self, mdp, gamma, num_iter):
        min_difference = 0.001
        state_values = {s: 0 for s in mdp.get_all_states()}

        for _ in range(num_iter):
            new_state_values = {
                state: self.get_new_state_value(mdp, state_values, state, gamma) for state in mdp.get_all_states()
            }

            assert isinstance(new_state_values, dict)

            diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())
            state_values = new_state_values

            if diff < min_difference:
                break

        return state_values

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
        mdp,
        step,
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
        state = f"s{str(state)}"

        state_values = self.find_state_values(mdp, self.gamma, self.num_iter)

        for i in range(time_limit):
            action = self.get_optimal_action(mdp, state_values, state, self.gamma)
            new_state, reward, done, _ = mdp.step(action)
            states.append(reversed_dict[int(state[1:])])

            actions.append(reversed_dict[int(action[1:])])
            states_cod.append(int(state[1:]))
            actions_cod.append(int(action[1:]))

            if not done:
                total_reward += (reward) * self.gamma ** (step + i) - self.penalty

            state = new_state
            state_coded = int(new_state[1:])

            if regime == "dynamic" and new_state in key_nodes:
                iteration += 1
                return (
                    states_cod,
                    states,
                    actions_cod,
                    actions,
                    total_reward,
                    iteration,
                    state_coded,
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
                    state_coded,
                    done,
                    step + i + 1,
                )

            if done:
                return (
                    states_cod + [state],
                    states + [reversed_dict[int(state[1:])]],
                    actions_cod,
                    actions,
                    total_reward,
                    iteration,
                    state_coded,
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
        mdp,
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
                reversed_dict=reversed_dict,
                mdp=mdp,
                time_limit=time_limit,
                step=step,
                regime=regime,
            )

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

    def apply(self, key_nodes, rewards, transition_probs, states_dict, reversed_dict, initial_state, mdp, regime):
        now = datetime.now()
        total_sessions = []

        for _ in range(self.n_learning_events):
            sessions = [
                self.generate_session(
                    initial_state=initial_state,
                    reversed_dict=reversed_dict,
                    key_nodes=key_nodes,
                    transition_probs=transition_probs,
                    rewards=rewards,
                    states_dict=states_dict,
                    mdp=mdp,
                    regime=regime,
                    time_limit=10**4,
                )
                for _ in range(self.n_sessions)
            ]

            total_sessions.extend(sessions)

            if (datetime.now() - now).seconds > self._time_break:
                print(
                    f"Достигнуто максимальное время выполнения {self._time_break} секунд. Переход к следующему методу."
                )
                break

        return total_sessions
