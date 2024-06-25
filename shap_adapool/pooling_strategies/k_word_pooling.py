import shap
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.typing import NDArray, ArrayLike
from typing import Callable, Iterable, Any
from functools import reduce, partial
from io import StringIO
import regex_spm
import argparse

from ..initializer import init
from ..pooler import unbatched_shap_value_pooler, two_element_sum
from ..token_concatenation import add_strings, k_word_concat
from ..plotting import save_plot


def main():
    parser = argparse.ArgumentParser(description='Script with integer argument.')

    parser.add_argument('k', type=int, help='k_value')

    args = parser.parse_args()

    k = args.k
    if k <= 0:
        raise ValueError("positive integer k required")


    init()
    # importing the shapley values from a pickle file to save some time

    with open('results/shap_values.pkl', 'rb') as f:
        shap_values = pickle.load(f)


    phrases, phrase_indices = k_word_concat(shap_values.data[0], k)

    values = unbatched_shap_value_pooler(shap_values.values[0],
                               phrase_indices,
                               two_element_sum, yield_last=True)

    # shap.plots.text(shap_values=)
    exp = shap._explanation.Explanation(values=np.array(list(values))[None, :],  # need to add batch dimension
                                        base_values=np.array([shap_values.base_values[0]]),
                                        data=((list(phrases),)))  # a 1-element tuple


    print(exp)
    single_sample_plotting_functions = [
        ("text", partial(shap.plots.text, display=False)),
    ]

    SAMPLE_IDX = 0
    for name, f in single_sample_plotting_functions:
        plot = f(shap_values=exp)
        save_plot(plot, name + "_pooling")


if __name__ == "__main__":
    main()