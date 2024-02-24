from typing import List, Any, Dict, Callable, Tuple, Literal
from functools import partial
from itertools import product, groupby
from dataclasses import dataclass

EvaluatorOutput = Any
Evaluator = Callable[..., EvaluatorOutput]
Metric = int | float
RawParameterValue = Any


@dataclass
class ParameterValue:
    """Simple dataclass to represent a parameter and its value."""
    name: str
    value: RawParameterValue

    def __hash__(self) -> int:
        # necessary to use the object as a key in a dictionary
        return hash((self.name, self.value))


def grid_search(evaluator: Evaluator,
                evaluator_kwargs: Dict[str, Any],
                parameter_grid: Dict[str, List[RawParameterValue]],
                metric_getter: Callable[[EvaluatorOutput], Metric],
                max_or_min: Literal["max", "min"] = "max",
                return_metrics: bool = False,
                return_metrics_grid: bool = False) -> ParameterValue |\
                                                      Tuple[ParameterValue, Metric] |\
                                                      Tuple[ParameterValue, Metric, Dict[ParameterValue, Metric]]:
    # create a partial call for a function with the parameters that are
    # not a part of the grid search:
    evaluator_ = partial(evaluator, **evaluator_kwargs)

    # turn the parameter dict into an array of `Parameter` objects:
    parameter_grid_ = [ParameterValue(name, value) for name, values in parameter_grid.items() for value in values]   

    # group the parameter grid by parameter name:
    grouped_parameter_grid = [tuple(v) for _, v in groupby(parameter_grid_, lambda p: p.name)]

    # create a list of all possible parameter combinations:
    parameter_combinations = list(product(*grouped_parameter_grid))
    # note: we need to unpack the `grouped_parameter_grid` above
    # because `product` expects iterables as arguments, so `[(..., ...), (..., ...)]`
    # would not work, we need to unpack it into `(..., ...), (..., ...)`

    # for each combination of parameters, call the evaluator and collect the metric:
    metrics = {}
    for parameter_combination in parameter_combinations:
        # call the partial function with the parameter value:
        parameter_values = {p.name: p.value for p in parameter_combination}
        evaluator_return_value = evaluator_(**parameter_values)

        # collect the metric from the evaluator return value:
        metric = metric_getter(evaluator_return_value)
        metrics[parameter_combination] = metric

    # select the best parameter combination based on the collected metrics:
    optimum_fn = lambda x: max(x, key=x.get) if max_or_min == "max" else lambda x: min(x, key=x.get)
    best = optimum_fn(metrics)

    match return_metrics, return_metrics_grid:
        case True, True:
            return best, metrics[best], metrics
        case True, False:
            return best, metrics[best]
        case False, True:
            return best, metrics
        case False, False:
            return best


if __name__ == "__main__":
    def evaluator(k, l, m): return k + l + m
    metric_getter = lambda x: x
    parameter_grid = {"k": [1, 2, 3], "l": [4, 5, 6], "m": [2, 1]}
    print(list(product(parameter_grid.values())))
    evaluator_kwargs = {}
    best = grid_search(evaluator, evaluator_kwargs, parameter_grid, metric_getter)
    print(best)
