from utils.flops_benchmark import add_flops_counting_methods


def init_decisioner_globals(train_func, test_func):
    global train, test
    train = train_func
    test = test_func


class Decisioner:
    def __init__(self, model_class, model_init_parameters):
        """
        model_init_parameters is a list of tuples containing:
            unique id (to distinguish different model data)
            model state dict
            model params
        """
        self._init_callable = model_class
        self._solutions = []
        for init_parameter in model_init_parameters:
            self._solutions.append(self._run_test(*init_parameter))

    def _run_test(self, id, state, model_params):
        model = self._init_callable(*model_params)
        model.load_state_dict(state)
        model = add_flops_counting_methods(model)
        model.start_flops_count()
        return id, test(model, printing=False), model.compute_average_flops_cost()

    def best_solution(self):
        sorted_solutions = list(reversed(sorted(self._solutions, key=lambda s: s[0])))
        return sorted_solutions[0]

    def worst_solution(self):
        sorted_solutions = sorted(self._solutions, key=lambda s: s[0])
        return sorted_solutions[0]

    def all_solutions(self):
        return self._solutions
