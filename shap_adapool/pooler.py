from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .types import ShapValueDtype

def two_element_sum(x, y): return x + y  # pylint: disable=multiple-statements


def unbatched_shap_value_pooler(values: NDArray[ShapValueDtype],
                                index_map: NDArray[np.int64],
                                aggregate_fn: Callable[[ShapValueDtype, ShapValueDtype], ShapValueDtype] = two_element_sum,
                                baseline_value: int = 0,
                                yield_last: bool = False):
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
            if yield_last:
                yield agg
            break
        if index_map[idx] != index_map[idx + 1]:
            yield agg
            agg = baseline_value  # reset accumulator


def shap_value_pooler(values: NDArray[ShapValueDtype],
                      index_map: NDArray[np.int64],
                      aggregate_fn: Callable[[ShapValueDtype, ShapValueDtype], ShapValueDtype] = two_element_sum,
                      baseline_value: int = 0,
                      yield_last: bool = False):
    """A variant of the `unbatched_shap_value_pooler` that can handle batched Shapley Values.
    See documentation for `unbatched_shap_value_pooler` for more information.
    
    > [!note]
    > The only difference in inputs here is that `values` and `index_map` are expected to be
    > 2D arrays, where the first dimension is the batch dimension.
    """
    if values.ndim != index_map.ndim:
        raise ValueError("Shapley Values and index_map must have the same number of dimensions")
    if values.ndim == 1:
        yield from unbatched_shap_value_pooler(values,
                                               index_map,
                                               aggregate_fn,
                                               baseline_value,
                                               yield_last)
    elif values.ndim > 2:
        raise ValueError("Shapley Values must be 1D (single sample) or 2D (batched) arrays")
    else:
        for idx, sample in values:
            yield np.array(unbatched_shap_value_pooler(sample,
                                                    index_map[idx],
                                                    aggregate_fn,
                                                    baseline_value,
                                                    yield_last))
