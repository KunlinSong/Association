import torch

from association.types import *


class Weight:
    def __new__(cls, size: Sequence[int], *, dtype: Optional[torch.dtype] = None) -> Self:
        weight = torch.nn.Parameter(torch.zeros(*size, dtype=dtype))
        torch.nn.init.xavier_uniform_(weight)
        return weight


class Bias:
    def __new__(cls,
                size: Sequence[int],
                *,
                dtype: Optional[torch.dtype] = None) -> Self:
        return torch.nn.Parameter(torch.zeros(*size, dtype=dtype))