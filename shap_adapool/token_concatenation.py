import numpy as np
import regex_spm
from numpy.typing import NDArray
from typing import Iterable, Callable, Protocol
from functools import reduce
from io import StringIO

from functools import partial

from .types import TokenDtype, IdxDtype, PhraseDtype

def add_strings(strings: Iterable[NDArray[TokenDtype]]) -> PhraseDtype:
    return str(reduce(np.char.add, strings))


class ConditionFunction(Protocol):

    def __call__(self, token: TokenDtype, idx: int, *args, **kwargs) -> bool:
        ...


def token_concat(array: NDArray[TokenDtype],
                 write_condition: ConditionFunction,
                 end_condition: ConditionFunction,
                 add_whitespace: bool = False,
                 *args,
                 **kwargs) -> (NDArray[PhraseDtype], NDArray[IdxDtype]):
    token_buffer = StringIO()
    concatenated_token_array = []
    index_map = []
    idx = 0
    for positional_idx, token in enumerate(array):
        index_map.append(idx)
        if write_condition(token, idx=positional_idx, *args, **kwargs):
            token_buffer.write(str(token))
            if add_whitespace:
                token_buffer.write(" ")
        if end_condition(token, idx=positional_idx, *args, **kwargs):
            token_buffer.write(str(token))
            concatenated_token_array.append(token_buffer.getvalue())
            token_buffer = StringIO()  # reset buffer
            idx += 1
    index_map.append(-1)  # indicates the end of the array
    return concatenated_token_array, index_map


def sentence_concat(array: NDArray[TokenDtype]) -> (NDArray[PhraseDtype], NDArray[IdxDtype]):

    def _end_condition(token: TokenDtype, idx: int) -> bool:
        match regex_spm.match_in(token):
            case r".*[\.:].*":  # if a token contains a period or a colon, it is the end of a sentence
                return True
            case _:
                return False

    def _write_condition(token: TokenDtype, idx: int) -> bool:
        return not _end_condition(token, idx)

    return partial(token_concat, write_condition=_write_condition, end_condition=_end_condition)(array)
