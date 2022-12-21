from __future__ import annotations

from datetime import datetime
import random

from numpy import argmax
from numpy import full as np_full
from numpy import max as np_max
from numpy import minimum as min_of_arrays
from numpy.random import choice as np_choice

from sklearn.preprocessing import normalize


def genetic_algorithm(
    muta,
    children,
    iters,
    gamma,
    key_nodes,
    rewards,
    transition_probs,
    states_dict,
    legal_actions,
    mdp,
    reversed_dict,
    initial_state,
    regime,
    penalty,
    end,
    short_path=None,
    time_break=10800,
):
    rew_genetic = -1e9

    gen = Genes(
        legal_actions=legal_actions,
        states_dict=states_dict,
        initial_state=initial_state,
        transition_probs=transition_probs,
        rewards=rewards,
        regime=regime,
        reversed_dict=reversed_dict,
        mdp=mdp,
        gamma=gamma,
        penalty=penalty,
        key_nodes=key_nodes,
        end=end,
        short_path=short_path,
        time_break=time_break,
    )
    best, rew = gen.apply(number_of_generations=iters, initial_population_size=children, mutation_probability=muta)

    if rew > rew_genetic:
        path_genetic = best
        rew_genetic = rew
        done_genetic = True

    if rew_genetic == -1e9:
        path_genetic = []
        done_genetic = False
    return path_genetic, rew_genetic, done_genetic


class RandPath:
    def __init__(self, gamma=0.9, penalty=0.0, short_path=None):

        self.gamma = gamma
        self.penalty = penalty
        self.counter = 0
        self._short_path = 1 if short_path is None else short_path

    @staticmethod
    def initialize_policy(legal_actions, iteration):
        policy = np_full([np_max(list(legal_actions[0])) + 1, np_max(list(legal_actions[0])) + 1], 0)

        for legal_action, bound_actions in legal_actions[iteration].items():
            for transition_action in bound_actions:
                policy[legal_action, transition_action] = 1

        policy = normalize(policy, axis=1, norm="l1")

        return policy

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

        for i in range(time_limit):
            policy = self.initialize_policy(legal_actions, iteration)
            action = np_choice(np_max(list(legal_actions[0])) + 1, p=policy[state])
            new_state, reward, done, _ = mdp.step(f"a{str(action)}")

            states.append(reversed_dict[state])
            actions.append(reversed_dict[action])
            states_cod.append(state)
            actions_cod.append(action)

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

    def apply(
        self,
        key_nodes,
        rewards,
        transition_probs,
        states_dict,
        legal_actions,
        mdp,
        reversed_dict,
        regime,
        initial_state,
    ):
        sessions = self.generate_session(
            initial_state=initial_state,
            reversed_dict=reversed_dict,
            key_nodes=key_nodes,
            transition_probs=transition_probs,
            rewards=rewards,
            states_dict=states_dict,
            legal_actions=legal_actions,
            mdp=mdp,
            regime=regime,
            time_limit=10**4,
        )

        return sessions


class Genes:
    def __init__(
        self,
        legal_actions,
        states_dict,
        initial_state,
        transition_probs,
        rewards,
        regime,
        reversed_dict,
        mdp,
        end,
        gamma=0.9,
        penalty=0.0,
        key_nodes=None,
        short_path=None,
        time_break=10800,
    ):

        self.end = end
        self.gamma = gamma
        self.penalty = penalty
        self.rewards = rewards
        self.states_dict = states_dict
        self.key_nodes = key_nodes
        self.mdp = mdp
        self.transition_probs = transition_probs
        self.initial_state = initial_state
        self.reversed_dict = reversed_dict
        self.legal_actions = legal_actions
        self.regime = regime
        self.max_iter = np_max(list(states_dict))
        self._short_path = 1 if short_path is None else short_path
        self._time_break = time_break

    def crossover(self, first_one, second_one):
        new_a, new_b = [], []
        possible_positions_first, possible_positions_second = (
            [i for i, x in enumerate(first_one) if x == self.key_nodes[0]],
            [i for i, x in enumerate(second_one) if x == self.key_nodes[0]],
        )

        can_cross = bool((len(possible_positions_first) > 0) & (len(possible_positions_second) > 0))

        if can_cross:
            possible_cross_points = min_of_arrays(len(possible_positions_first), len(possible_positions_second))
            possible_positions_first, possible_positions_second = (
                possible_positions_first[:possible_cross_points],
                possible_positions_second[:possible_cross_points],
            )

            cut_pos = (
                random.randint(0, len(possible_positions_first) - 1) if len(possible_positions_first) > 1 else 0
            )

            temp_one, temp_two = first_one[1:-1], second_one[1:-1]
            cut_one, cut_two = (
                temp_one[possible_positions_first[cut_pos] :],
                temp_two[possible_positions_second[cut_pos] :],
            )
            cut_a, cut_b = (
                temp_one[: possible_positions_first[cut_pos]],
                temp_two[: possible_positions_second[cut_pos]],
            )

            new_a, new_b = cut_a + cut_two, cut_b + cut_one
            new_a.insert(0, first_one[0])
            new_b.insert(0, first_one[0])
            new_a.append(first_one[-1])
            new_b.append(first_one[-1])

            crossover_threshold = min(len(first_one), len(second_one))
            crossover_threshold = int(0.5 * crossover_threshold)

        return new_a, new_b, can_cross

    def fitness(self, path):
        reward = 0.0

        if self.regime == "dynamic":
            kakoe_k = max(
                min(self.states_dict),
                1,
            )

            for step in range(1, len(path)):
                if path[step] != self.end:
                    reward += (
                        self.rewards[kakoe_k]
                        .get(f"s{str(self.states_dict[kakoe_k][path[step - 1]])}", {})
                        .get(f"a{str(self.states_dict[kakoe_k][path[step]])}", {})
                        .get(f"s{str(self.states_dict[kakoe_k][path[step]])}", 0.0)
                        * self.gamma ** (step - 1)
                    ) - self.penalty

                if path[step] == self.key_nodes[0]:
                    kakoe_k += 1
                if kakoe_k > self.max_iter:
                    kakoe_k -= 1

                kakoe_k = min(self.states_dict)

        elif self.regime == "static":
            kakoe_k = 0
            kakoe_s = 0

            for step in range(1, len(path)):
                if path[step] != self.end:
                    reward += (
                        self.rewards[kakoe_k]
                        .get(f"s{str(self.states_dict[kakoe_k][path[step - 1]])}", {})
                        .get(f"a{str(self.states_dict[kakoe_k][path[step]])}", {})
                        .get(f"s{str(self.states_dict[kakoe_k][path[step]])}", 0.0)
                        * self.gamma ** (step - 1)
                    ) - self.penalty

                if path[step] == self.key_nodes[0]:
                    kakoe_s += 1
                if kakoe_s >= self._short_path:
                    kakoe_k = -2

        return [path, reward]

    def mutate(self, route, probability):
        if random.random() < probability:
            path = RandPath(gamma=self.gamma, penalty=self.penalty)

            return path.apply(
                key_nodes=self.key_nodes,
                rewards=self.rewards,
                transition_probs=self.transition_probs,
                initial_state=self.initial_state,
                states_dict=self.states_dict,
                legal_actions=self.legal_actions,
                mdp=self.mdp,
                reversed_dict=self.reversed_dict,
                regime=self.regime,
            )[0]

        return route

    def create_new_generation(self, size):
        path = RandPath(gamma=self.gamma, penalty=self.penalty, short_path=self._short_path)

        return [
            next(
                path.apply(
                    key_nodes=self.key_nodes,
                    rewards=self.rewards,
                    transition_probs=self.transition_probs,
                    initial_state=self.initial_state,
                    states_dict=self.states_dict,
                    legal_actions=self.legal_actions,
                    mdp=self.mdp,
                    reversed_dict=self.reversed_dict,
                    regime=self.regime,
                )
            )
            for _ in range(size)
        ]

    def apply(self, number_of_generations, initial_population_size, mutation_probability):
        now = datetime.now()
        population_fitness = []
        new_gen = self.create_new_generation(initial_population_size)

        for _ in range(number_of_generations):
            population_fitness = sorted(
                [self.fitness(new_gen[k]) for k in range(len(new_gen))], key=lambda x: x[1], reverse=True
            )

            lena = len(population_fitness) // 1.5
            new_gen_cut = [x[0] for x in population_fitness[: int(lena)]]
            random.shuffle(new_gen_cut)

            exceptions = []
            next_gen = []

            for gene_idx, _ in enumerate(new_gen_cut):
                for gene_iterator, _ in enumerate(new_gen_cut):
                    if (
                        gene_iterator != gene_idx
                        and gene_idx not in set(exceptions)
                        and gene_iterator not in set(exceptions)
                    ):
                        cross_a, cross_b, breed_status = self.crossover(
                            new_gen_cut[gene_idx], new_gen_cut[gene_iterator]
                        )

                        if breed_status:
                            next_gen.extend((cross_a, cross_b))
                            exceptions.extend((gene_idx, gene_iterator))

                    if len(next_gen) > len(new_gen_cut) * 0.8:
                        break
                else:
                    continue
                break

            pop_inc = abs(initial_population_size - len(next_gen))
            next_gen.extend([next(x) for x in population_fitness[:pop_inc]])

            for idx, _ in enumerate(next_gen):
                next_gen[idx] = self.mutate(next_gen[idx], mutation_probability)

            new_gen = next_gen

            if (datetime.now() - now).seconds > self._time_break:
                print(
                    f"Достигнуто максимальное время выполнения {self._time_break} секунд. Переход к следующему методу."
                )
                break

        done_population_fitness = [item for item in population_fitness if item[0][-1] == self.end]
        best = done_population_fitness[argmax([i[-1] for i in done_population_fitness])]

        return best[:2]
