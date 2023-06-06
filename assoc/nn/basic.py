from typing import overload

import torch

from assoc.types import *

__all__ = ['Bias', 'MapDense', 'Weight']


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
        return torch.nn.Parameter(torch.randn(*size, dtype=dtype) / 100)


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


class MapDense(torch.nn.Module):
    """Dense layer with map units.
    
    Attributes:
        map_units (int): Number of map units.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        feature_last (bool): Whether the last dimension of input is features.
        bias (bool): Whether to use bias.
        dtype (torch.dtype): Data type of weight and bias.
    """

    def __init__(self,
                 map_units: int,
                 in_features: int,
                 out_features: int,
                 *,
                 feature_last: bool = True,
                 bias: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.weight = Weight(in_features, out_features, dtype=dtype)
        self.bias = Bias(out_features, dtype=dtype) if bias else None
        self.map_units = map_units
        self.in_features = in_features
        self.out_features = out_features
        self.feature_last = feature_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.dim()
        assert (shape[-2:] == (self.map_units, self.in_features)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.map_units}, {self.in_features}), but received {shape}'
        )
        y = torch.einsum('...ij,jk->...ik', x, self.weight)
        return y + self.bias if self.bias is not None else y
