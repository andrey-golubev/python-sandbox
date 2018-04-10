import math
import copy
import numpy as np
import torch
from utils.flops_benchmark import add_flops_counting_methods


def init_optimizer_globals(train_func, test_func):
    global train, test
    train = train_func
    test = test_func


class Optimizer:
    def __init__(self, model_initializer, baseline_acc, params, optimization_alias):
        self._init_callable = model_initializer
        self._baseline_acc = baseline_acc
        self._baseline_params = params
        self._best_score = (0.0, math.inf)
        self._best_params = None
        self._best_state_dict = None
        self._epsilon = 0.07
        self._optimization_func_aliases = {
            0: self._optimize_impl0,
            1: self._optimize_impl1
        }
        self._optimize_impl = self._optimization_func_aliases[optimization_alias]

    def _decision_function(self, score1, score2):
        b_acc, _ = self._baseline_acc
        (acc1, flops1), (acc2, flops2) = score1, score2
        bad1, bad2 = False, False
        if math.abs(acc1 - b_acc) > self._epsilon:
            bad1 = True
        if math.abs(acc2 - b_acc) > self._epsilon:
            bad2 = True
        if bad1 and bad2:
            return self._best_score, self._best_params
        if bad1 or flops2 < flops1:
            return score2
        if bad2 or flops1 < flops2:
            return score1

    def _run_test(self, state, model_params):
        # print(model_params)
        model = self._init_callable(*model_params)
        model.load_state_dict(state)
        # print('After load')
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

    def _optimize_impl0(self, optimization_state, optimization_params, opt_layer, chg_layer, param_index):
        opt_weight_s, opt_bias_s, chg_weight_s = Optimizer._get_names(opt_layer, chg_layer)
        indices = list(range(0, *optimization_state[opt_bias_s].size()))

        # 1: calculate metrics per index
        metric_per_param = []
        for index in indices:
            state = Optimizer._update_state(
                optimization_state,
                len(indices),
                [index],
                (opt_layer, chg_layer)
            )

            params = list(optimization_params)
            params[param_index] = 1
            acc, flops = self._run_test(state, params)
            metric_per_param.append((index, acc, flops))

        # 2: sort metrics
        metric_per_param = list(reversed(sorted(metric_per_param, key=lambda e: e[1])))

        # 3: reduction
        last_viable_params = optimization_params
        last_viable_state = optimization_state
        for limit in reversed(indices[1:len(indices)]):
            sln_indices = [v[0] for v in metric_per_param[:limit]]
            state = Optimizer._update_state(
                optimization_state,
                len(indices),
                sln_indices,
                (opt_layer, chg_layer)
            )
            params = list(optimization_params)
            params[param_index] = len(sln_indices)
            acc, flops = self._run_test(state, params)
            if (self._baseline_acc - acc) > self._epsilon:
                break
            last_viable_params = params
            last_viable_state = state
        return last_viable_state, last_viable_params

    def _optimize_impl1(self, optimization_state, optimization_params, opt_layer, chg_layer, param_index):
        return optimization_state, optimization_params

    def optimize(self, model_state_dict, optimization_input):
        optimization_params = self._baseline_params
        optimization_state = model_state_dict
        for opt_in in optimization_input:
            optimization_state, optimization_params = self._optimize_impl(
                optimization_state,
                optimization_params,
                *opt_in)

        return optimization_params, optimization_state
