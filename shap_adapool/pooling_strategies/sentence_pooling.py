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

from ..initializer import init
from ..pooler import unbatched_shap_value_pooler, two_element_sum
from ..token_concatenation import add_strings, sentence_concat
from ..plotting import save_plot


def main():

    init()
    # importing the shapley values from a pickle file to save some time

    with open('shap.pkl', 'rb') as f:
        shap_values = pickle.load(f)


    sentences, sentence_indices = sentence_concat(shap_values.data[0])

    values = unbatched_shap_value_pooler(shap_values.values[0],
                               sentence_indices,
                               two_element_sum)

    # shap.plots.text(shap_values=)
    exp = shap._explanation.Explanation(values=np.array(list(values))[None, :],  # need to add batch dimension
                                        base_values=np.array([shap_values.base_values[0]]),
                                        data=((list(sentences),)))  # a 1-element tuple

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
