import numpy as np
import regex_spm
from numpy.typing import NDArray
from typing import Iterable, Callable
from functools import reduce
from io import StringIO

from functools import partial

from .types import TokenDtype, IdxDtype, PhraseDtype

def add_strings(strings: Iterable[NDArray[TokenDtype]]) -> PhraseDtype:
    return str(reduce(np.char.add, strings))


def token_concat(array: NDArray[TokenDtype],
                 write_condition: Callable[[TokenDtype], bool],
                 end_condition: Callable[[TokenDtype], bool]) -> (NDArray[PhraseDtype], NDArray[IdxDtype]):
    token_buffer = StringIO()
    concatenated_token_array = []
    index_map = []
    idx = 0
    for token in array:
        index_map.append(idx)
        if write_condition(token):
            token_buffer.write(token)
        if end_condition(token):
            token_buffer.write(token)
            concatenated_token_array.append(token_buffer.getvalue())
            token_buffer = StringIO()  # reset buffer
            idx += 1
    index_map.append(-1)  # indicates the end of the array
    return concatenated_token_array, index_map


def sentence_concat(array: NDArray[TokenDtype]) -> (NDArray[PhraseDtype], NDArray[IdxDtype]):

    def _end_condition(token: TokenDtype) -> bool:
        match regex_spm.match_in(token):
            case r".*[\.:].*":  # if a token contains a period or a colon, it is the end of a sentence
                return True
            case _:
                return False

    def _write_condition(token: TokenDtype) -> bool:
        return not _end_condition(token)

    return partial(token_concat, write_condition=_write_condition, end_condition=_end_condition)(array)
