import torch

from assoc.types import *
from assoc.nn.basic import *

__all__ = ['GraphGRUCell', 'GraphLSTMCell', 'GraphRNNCell']


def _get_gate_params(
    input_size: int, hidden_size: int, dtype: torch.dtype
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    """Get gate parameters.
    
    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        dtype (torch.dtype): Data type."""
    return (Weight(input_size, hidden_size,
                   dtype=dtype), Weight(hidden_size, hidden_size, dtype=dtype),
            Bias(hidden_size, dtype=dtype))


class GraphLSTMCell(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 input_size: int,
                 hidden_size: int,
                 *,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.nodes = nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_last = feature_last
        self.dtype = dtype
        for gate in ('i', 'f', 'o', 'g'):
            w, u, b = _get_gate_params(input_size, hidden_size, dtype)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.dim()
        assert (shape[-2:] == (self.nodes, self.input_size)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.input_size}), but received {shape}')

        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)
        if state is None:
            h = torch.zeros(x.shape[0],
                            self.nodes,
                            self.hidden_size,
                            dtype=self.dtype, device=self.w_i.d)
            c = torch.zeros(x.shape[0],
                            self.nodes,
                            self.hidden_size,
                            dtype=self.dtype)
        else:
            h, c = state if is_batched else (torch.unsqueeze(state[0], 0),
                                             torch.unsqueeze(state[1], 0))

        i = torch.sigmoid(
            torch.einsum('bmi,ih->bmh', x, self.w_i) +
            torch.einsum('bmi,ih->bmh', h, self.u_i) + self.b_i)
        f = torch.sigmoid(
            torch.einsum('bmi,ih->bmh', x, self.w_f) +
            torch.einsum('bmi,ih->bmh', h, self.u_f) + self.b_f)
        o = torch.sigmoid(
            torch.einsum('bmi,ih->bmh', x, self.w_o) +
            torch.einsum('bmi,ih->bmh', h, self.u_o) + self.b_o)
        g = torch.tanh(
            torch.einsum('bmi,ih->bmh', x, self.w_g) +
            torch.einsum('bmi,ih->bmh', h, self.u_g) + self.b_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h, c) if is_batched else (torch.squeeze(h, 0),
                                          torch.squeeze(c, 0))


class GraphGRUCell(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 input_size: int,
                 hidden_size: int,
                 *,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.nodes = nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_last = feature_last
        self.dtype = dtype
        for gate in ('r', 'z', 'n'):
            w, u, b = _get_gate_params(input_size, hidden_size, dtype)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(self,
                x: torch.Tensor,
                *,
                state: Optional[tuple[torch.Tensor]] = None) -> torch.Tensor:
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.dim()
        assert (shape[-2:] == (self.nodes, self.input_size)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.input_size}), but received {shape}')

        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)
        if state is None:
            h = torch.zeros(x.shape[0],
                            self.nodes,
                            self.hidden_size,
                            dtype=self.dtype)
        else:
            h = state[0] if is_batched else torch.unsqueeze(state[0], 0)

        r = torch.sigmoid(
            torch.einsum('bmi,ih->bmh', x, self.w_r) +
            torch.einsum('bmi,ih->bmh', h, self.u_r) + self.b_r)
        z = torch.sigmoid(
            torch.einsum('bmi,ih->bmh', x, self.w_z) +
            torch.einsum('bmi,ih->bmh', h, self.u_z) + self.b_z)
        n = torch.tanh(
            torch.einsum('bmi,ih->bmh', x, self.w_n) +
            r * torch.einsum('bmi,ih->bmh', h, self.u_n) + self.b_n)
        h = (1 - z) * n + z * h
        return (h, ) if is_batched else (torch.squeeze(h, 0), )


class GraphRNNCell(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 input_size: int,
                 hidden_size: int,
                 *,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.nodes = nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_last = feature_last
        self.dtype = dtype
        w, u, b = _get_gate_params(input_size, hidden_size, dtype)
        self.w = w
        self.u = u
        self.b = b

    def forward(self,
                x: torch.Tensor,
                *,
                state: Optional[tuple[torch.Tensor]] = None) -> torch.Tensor:
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.dim()
        assert (shape[-2:] == (self.nodes, self.input_size)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.input_size}), but received {shape}')

        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)
        if state is None:
            h = torch.zeros(x.shape[0],
                            self.nodes,
                            self.hidden_size,
                            dtype=self.dtype)
        else:
            h = state[0] if is_batched else torch.unsqueeze(state[0], 0)

        h = torch.tanh(
            torch.einsum('bmi,ih->bmh', x, self.w) +
            torch.einsum('bmi,ih->bmh', h, self.u) + self.b)
        return (h, ) if is_batched else (torch.squeeze(h, 0), )


RNNCellModule = Union[GraphGRUCell, GraphLSTMCell, GraphRNNCell]