from typing import overload

import torch

from assoc.types import *

__all__ = ['Bias', 'Weight']


class Weight:
    """Weight initializer."""

    @overload
    def __new__(cls, *size: int, dtype: Optional[torch.dtype] = None) -> Self:
        ...

    @overload
    def __new__(cls,
                size: Sequence[int],
                *,
                dtype: Optional[torch.dtype] = None) -> Self:
        ...

    def __new__(cls,
                *size: Union[int, Sequence[int]],
                dtype: Optional[torch.dtype] = None) -> Self:
        if len(size) == 1 and isinstance(size[0], Sequence):
            size = size[0]
        weight = torch.nn.Parameter(torch.zeros(*size, dtype=dtype))
        torch.nn.init.xavier_uniform_(weight)
        return weight


class Bias:
    """Bias initializer."""

    @overload
    def __new__(cls, *size: int, dtype: Optional[torch.dtype] = None) -> Self:
        ...

    @overload
    def __new__(cls,
                size: Sequence[int],
                *,
                dtype: Optional[torch.dtype] = None) -> Self:
        ...

    def __new__(cls,
                *size: Union[int, Sequence[int]],
                dtype: Optional[torch.dtype] = None) -> Self:
        if len(size) == 1 and isinstance(size[0], Sequence):
            size = size[0]
        return torch.nn.Parameter(torch.zeros(*size, dtype=dtype))