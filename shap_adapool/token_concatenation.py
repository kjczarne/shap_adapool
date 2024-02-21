import numpy as np
import regex_spm
from numpy.typing import NDArray
from typing import Iterable
from functools import reduce
from io import StringIO

from .types import TokenDtype, IdxDtype, PhraseDtype

def add_strings(strings: Iterable[NDArray[TokenDtype]]) -> PhraseDtype:
    return str(reduce(np.char.add, strings))


def sentence_concat(array: NDArray[TokenDtype]) -> (NDArray[PhraseDtype], NDArray[IdxDtype]):
    sentences = []
    token_buffer = StringIO()
    index_map = []
    idx = 0
    for token in array:
        index_map.append(idx)
        match regex_spm.match_in(token):
            case r".*[\.:].*":  # if a token contains a period or a colon, it is the end of a sentence
            # case r"\D+[\.:]\D+":
                token_buffer.write(token)
                sentences.append(token_buffer.getvalue())
                token_buffer = StringIO()  # reset buffer
                idx += 1
            case _:
                token_buffer.write(token)
    index_map.append(-1)  # indicates the end of the array
    return sentences, index_map

def k_word_concat(array: NDArray[TokenDtype], k: int) -> (NDArray[PhraseDtype], NDArray[IdxDtype]):
    sentences = []
    token_buffer = StringIO()
    index_map = []
    idx = 0
    i = 1
    arrayLength = len(array)
    for token in array:
        index_map.append(idx)
        if i % k == 0 or i == arrayLength:
            token_buffer.write(token)
            sentences.append(token_buffer.getvalue())
            token_buffer = StringIO()  # reset buffer
            idx += 1
        else:
            token_buffer.write(token)
        i += 1
    index_map.append(-1)  # indicates the end of the array
    return sentences, index_map