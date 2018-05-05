import math
import copy
import numpy as np
import itertools
import random
import torch

from utils.flops_benchmark import add_flops_counting_methods
from model_decisioner import Decisioner
from model_qlearner import Agent, convert


def init_optimizer_globals(train_data, test_func):
    global train, train_optimizer, optimizer_params, fine_tune_epochs, test
    train, train_optimizer, optimizer_params, fine_tune_epochs = train_data
    test = test_func

    # Q-learning things:
    global reward_coeff
    reward_coeff = 0.5

    global expected_compression_ratio
    expected_compression_ratio = 0.5

    global q_train_episodes
    q_train_episodes = 100

    global least_iter_num
    least_iter_num = 10


def _find_subsets(S, subset_size):
    return set(itertools.combinations(S, subset_size))


class Optimizer:
    def __init__(self, model_initializer, baseline_acc, baseline_flops, params, method_index, epsilon=0.07):
        self._init_callable = model_initializer
        self._baseline_acc = baseline_acc
        self._baseline_flops = baseline_flops
        self._baseline_params = params
        self._best_score = (0.0, math.inf)
        self._best_params = None
        self._best_state_dict = None
        self._epsilon = epsilon
        self._optimization_func_aliases = {
            0: self._optimize_impl0,
            1: self._optimize_impl1,
            2: self._optimize_impl2,
            3: self._logging_optimization0
        }
        self._optimize_impl = self._optimization_func_aliases[method_index]

    def _run_test(self, state, model_params):
        model = self._init_callable(*model_params)
        model.load_state_dict(state)
        model = add_flops_counting_methods(model)
        model.start_flops_count()
        return test(model, printing=False), model.compute_average_flops_cost()

    @staticmethod
    def _reduce_state_opt(orig_state, orig_size, indices, size_pos):
        new_size = list(orig_state.size())
        new_size[size_pos] = int(new_size[size_pos] / orig_size * len(indices))
        state = torch.FloatTensor(*new_size)
        for i, target_i in enumerate(indices):
            state[i] = orig_state[target_i]
        return state

    @staticmethod
    def _reduce_state_chg(orig_state, orig_size, indices, size_pos):
        new_size = list(orig_state.size())
        new_size[size_pos] = int(new_size[size_pos] / orig_size * len(indices))
        state = torch.FloatTensor(*new_size)
        for row, orig_row in zip(state, orig_state):
            for i, target_i in enumerate(indices):
                row[i] = orig_row[target_i]
        return state

    @staticmethod
    def _get_names(opt_layer, chg_layer):
        return '{0}.weight'.format(opt_layer), '{0}.bias'.format(opt_layer), '{0}.weight'.format(chg_layer)

    @staticmethod
    def _update_state(orig_state, orig_indices_size, indices, layer_names):
        opt_layer, chg_layer = layer_names
        opt_weight_s, opt_bias_s, chg_weight_s = Optimizer._get_names(opt_layer, chg_layer)
        state = copy.deepcopy(orig_state)
        state[opt_weight_s] = Optimizer._reduce_state_opt(
            state[opt_weight_s],
            orig_indices_size,
            indices,
            0
        )
        state[opt_bias_s] = Optimizer._reduce_state_opt(
            state[opt_bias_s],
            orig_indices_size,
            indices,
            0
        )
        state[chg_weight_s] = Optimizer._reduce_state_chg(
            state[chg_weight_s],
            orig_indices_size,
            indices,
            1
        )
        return state

    @staticmethod
    def _create_state_params(orig_state, orig_indices_size, indices, layer_names, model_params, p_index):
        state = Optimizer._update_state(
            orig_state,
            orig_indices_size,
            indices,
            layer_names
        )
        params = list(model_params)
        params[p_index] = len(indices)
        return state, params

    def _fine_tune(self, model_state, model_params, epochs):
        model = self._init_callable(*model_params)
        model.load_state_dict(model_state)
        model_optimizer = train_optimizer(model.parameters(), **optimizer_params)
        for _ in range(epochs):
            train(model, model_optimizer)
        return model.state_dict(), model_params

    def optimize(self, model_state_dict, optimization_input):
        return self._q_learning_impl(model_state_dict, optimization_input)
        # optimization_params = self._baseline_params
        # optimization_state = model_state_dict
        # for opt_in in optimization_input:
        #     # print('Optimization Step:', opt_in)

        #     # run reduction algorithm
        #     optimization_state, optimization_params = self._optimize_impl(
        #         optimization_state,
        #         optimization_params,
        #         *opt_in)

        #     # fine-tune after reduction
        #     optimization_state, optimization_params = self._fine_tune(
        #         optimization_state,
        #         optimization_params,
        #         fine_tune_epochs)

        # return optimization_params, optimization_state

    def _optimize_impl0(self, optimization_state, optimization_params, opt_layer, chg_layer, param_index):
        """
        Optimization algo 1: heuristic
        """
        indices = list(range(0, self._baseline_params[param_index]))

        # 1: calculate metrics per index
        metric_per_param = []
        for index in indices:
            state, params = Optimizer._create_state_params(
                optimization_state, 
                len(indices), 
                [index],
                (opt_layer, chg_layer),
                optimization_params,
                param_index
            )
            acc, flops = self._run_test(state, params)
            metric_per_param.append((index, acc, flops))

        # 2: sort metrics
        metric_per_param = list(reversed(sorted(metric_per_param, key=lambda e: e[1])))

        # 3: reduction
        last_viable_params = optimization_params
        last_viable_state = optimization_state
        for limit in reversed(indices[1:len(indices)]):
            sln_indices = [v[0] for v in metric_per_param[:limit]]
            state, params = Optimizer._create_state_params(
                optimization_state, 
                len(indices), 
                sln_indices,
                (opt_layer, chg_layer),
                optimization_params,
                param_index
            )

            acc, flops = self._run_test(state, params)
            if (self._baseline_acc - acc) > self._epsilon:
                break
            last_viable_params = params
            last_viable_state = state
        return last_viable_state, last_viable_params

    def _optimize_impl1(self, optimization_state, optimization_params, opt_layer, chg_layer, param_index):
        """
        Optimization algo 2: greedy

        Until a viable solution found, try to increase depth as much as possible
        Once found, return that solution -> this is the best possible
        A solution is a pair (acc, flops) which is the best one found during iteration with K elements reduction
            Best pair is found by decision_function
        A viable solution is the one that gives baseline_acc - sln_acc <= epsilon
        """
        indices = list(range(0, self._baseline_params[param_index]))

        viable_state, viable_params = optimization_state, optimization_params
        # found_viable_sln = False
        for subset_size in range(1, len(indices)):
            # if found_viable_sln:
            #     break

            all_index_subsets = list(_find_subsets(indices, subset_size))
            potential_solutions = []
            for i, sln_indices in enumerate(all_index_subsets):
                state, params = Optimizer._create_state_params(
                    optimization_state, 
                    len(indices), 
                    sln_indices,
                    (opt_layer, chg_layer),
                    optimization_params,
                    param_index
                )
                potential_solutions.append((i, state, params))

            decisioner = Decisioner(self._init_callable, potential_solutions)
            best_id, best_acc, _ = decisioner.best_solution()
            if (self._baseline_acc - best_acc) <= self._epsilon:
                # found_viable_sln = True
                viable_state, viable_params = Optimizer._create_state_params(
                    optimization_state, 
                    len(indices), 
                    all_index_subsets[best_id],
                    (opt_layer, chg_layer),
                    optimization_params,
                    param_index
                )
                break

        return viable_state, viable_params

    def _optimize_impl2(self, optimization_state, optimization_params, opt_layer, chg_layer, param_index):
        """
        Optimization algo 3: greedy 2

        While a viable solution exists, try to decrease the depth as much as possible
        Once there's a non-viable solution, return the last viable -> this is the best possible
        A solution is a pair (acc, flops) which is the best one found during iteration with K elements reduction
            Best pair is found by decision_function
        A viable solution is the one that gives baseline_acc - sln_acc <= epsilon
        Additionally, iteration's limit is presented
        """
        max_iterations = 50
        indices = list(range(0, self._baseline_params[param_index]))

        viable_state, viable_params = optimization_state, optimization_params
        # found_nonviable_sln = False
        for subset_size in list(reversed(range(1, len(indices)))):
            # if found_nonviable_sln:
                # break

            all_index_subsets = list(_find_subsets(indices, subset_size))
            potential_solutions = []
            for i, sln_indices in enumerate(all_index_subsets):
                state, params = Optimizer._create_state_params(
                    optimization_state,
                    len(indices),
                    sln_indices,
                    (opt_layer, chg_layer),
                    optimization_params,
                    param_index
                )
                potential_solutions.append((i, state, params))

            if len(potential_solutions) > max_iterations:
                potential_solutions = random.sample(potential_solutions, max_iterations)

            decisioner = Decisioner(self._init_callable, potential_solutions)
            best_id, best_acc, _ = decisioner.best_solution()
            if (self._baseline_acc - best_acc) > self._epsilon:
                # found_nonviable_sln = True
                break

            viable_state, viable_params = Optimizer._create_state_params(
                optimization_state,
                len(indices),
                all_index_subsets[best_id],
                (opt_layer, chg_layer),
                optimization_params,
                param_index
            )

        return viable_state, viable_params

    def _logging_optimization0(self, optimization_state, optimization_params, opt_layer, chg_layer, param_index):
        """
        Logging algorithm

        While a viable solution exists, try to decrease the depth as much as possible
        Once there's a non-viable solution, return the last viable -> this is the best possible
        A solution is a pair (acc, flops) which is the best one found during iteration with K elements reduction
            Best pair is found by decision_function
        A viable solution is the one that gives baseline_acc - sln_acc <= epsilon
        Additionally, iteration's limit is presented
        """
        import csv
        max_iterations = 50
        viable_state, viable_params = optimization_state, optimization_params
        indices = list(range(0, self._baseline_params[param_index]))
        with open('logs_{0}_1.csv'.format(opt_layer), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Size', 'Accuracy', 'FLOPS'])

            # found_nonviable_sln = False
            for subset_size in list(reversed(range(1, len(indices)))):
                all_index_subsets = list(_find_subsets(indices, subset_size))
                potential_solutions = []
                for i, sln_indices in enumerate(all_index_subsets):
                    state, params = Optimizer._create_state_params(
                        optimization_state,
                        len(indices),
                        sln_indices,
                        (opt_layer, chg_layer),
                        optimization_params,
                        param_index
                    )
                    potential_solutions.append((i, state, params))

                if len(potential_solutions) > max_iterations:
                    potential_solutions = random.sample(potential_solutions, max_iterations)

                decisioner = Decisioner(self._init_callable, potential_solutions)
                all_solutions = decisioner.all_solutions()
                for _, accuracy, flops in all_solutions:
                    writer.writerow([subset_size, accuracy, flops])

        return viable_state, viable_params

    def _change_env(self, model_state, model_params, optimization_input, new_env_state):
        start = 0
        for opt_layer, chg_layer, opt_index in optimization_input:
            layer_size = self._baseline_params[opt_index]

            # create weights to include
            included_weights = [i for i, v in enumerate(new_env_state[start:start+layer_size]) if v == 1]
            # update model params and state
            model_state, model_params = Optimizer._create_state_params(
                model_state,
                layer_size,
                included_weights,
                (opt_layer, chg_layer),
                model_params,
                opt_index
            )
            start += layer_size
        return model_state, model_params

    def _q_learning_impl(self, model_state_dict, optimization_input):
        state_size = 0  # environment state size

        # preparation:
        for _, _, opt_index in optimization_input:
            state_size += self._baseline_params[opt_index]

        agent = Agent(state_size)

        for _ in range(q_train_episodes):  # recurrent loop to train the agent
            env_state = np.array([1]*state_size)  # environment state: initial - all weights are enabled
            done = False

            local_state = model_state_dict
            local_params = self._baseline_params

            iteration = 0
            while not done or iteration < least_iter_num:  # optimization loop
                # perform action
                indices = agent.act(env_state)

                # see the state transition and reward:
                # 1. create new state -- apply action
                next_state = env_state
                for i, flag in enumerate(indices):
                    if flag > 0:  # any positive number - switch feature map
                        next_state[i] = convert(env_state[i])

                # 2. collect statistics -- see consequences
                local_state, local_params = self._change_env(
                    local_state,
                    local_params,
                    optimization_input,
                    next_state
                )
                acc, flops = self._run_test(local_state, local_params)

                # 3. calculate reward -- measure consequences
                flops_ratio = (self._baseline_flops - flops) / self._baseline_flops
                reward_part = reward_coeff * flops_ratio
                punishment_part = (1.0 - reward_coeff) * (acc - self._baseline_acc)
                reward = reward_part + punishment_part

                # 4. change status
                if (self._baseline_acc - acc) > self._epsilon or flops_ratio >= expected_compression_ratio:
                    done = True

                # remember occasion
                agent.remember(env_state, indices, reward, next_state, done)

                # update environment
                env_state = next_state

                iteration += 1
                # exit if finished
                if done:
                    break

            # train the agent
            agent.replay(27)

        # make decision
        env_state = np.array([1]*state_size)
        indices = agent.act(env_state)
        for i, flag in enumerate(indices):
            if flag > 0:  # any positive number - switch feature map
                env_state[i] = convert(env_state[i])

        # update optim_state, optim_params
        optimization_state, optimization_params = self._change_env(
            model_state_dict,
            self._baseline_params,
            optimization_input,
            env_state
        )
        optimization_state, optimization_params = self._fine_tune(
            optimization_state,
            optimization_params,
            fine_tune_epochs
        )

        return optimization_params, optimization_state
