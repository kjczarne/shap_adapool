from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .types import ShapValueDtype

def two_element_sum(x, y): return x + y  # pylint: disable=multiple-statements


def shap_value_pooler(values: NDArray[ShapValueDtype],
                      index_map: NDArray[np.int64],
                      aggregate_fn: Callable[[ShapValueDtype, ShapValueDtype], ShapValueDtype] = two_element_sum,
                      baseline_value: int = 0,
                      yieldLast: bool = False):
    """Aggregates Shapley Values based on the index map. The index map indicates
    which tokens belong to the same sentence/phrase. The aggregator function is applied,
    between the aggregator and the subsequent token, until the index changes.
    By default the aggregator function is just a simple two-element-sum.
    
    Example:
    Consider the following index map:
        [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    Now consider the following token inputs:
        [my, name, is, john, smith, and, i, am, 20, years, old]
    And the calculated Shapley Values for these:
        [0.03, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01]
    Using the default two-element-sum aggregator function, the result would be:
        [0.03 + 0.01 + 0.02, 0.01 + 0.01, 0.02 + 0.01 + 0.02 + 0.02 + 0.01]
        [0.06, 0.02, 0.08]
    
    > [!note]
    > This function does not return the tokens, nor are the tokens an input to this function.
    > This pooler is meant only for the values themselves and the way you implement your token
    > pooler is up to you, as long as you have a list of indices indicating which tokens belong
    > to which sentence/phrase.
    """
    agg = baseline_value
    for idx, value in enumerate(values):
        agg = aggregate_fn(value, agg)
        if index_map[idx + 1] == -1:
            if yieldLast:
                yield agg
            break
        if index_map[idx] != index_map[idx + 1]:
            yield agg
            agg = baseline_value  # reset accumulator
