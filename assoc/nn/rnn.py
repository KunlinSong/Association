import torch

from assoc.types import *
from assoc.nn.basic import *


def _get_gate_params(
    input_size: int, hidden_size: int, dtype: torch.dtype
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    """Get gate parameters.
    
    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        dtype (torch.dtype): Data type."""
    return (Weight(input_size, hidden_size, dtype=dtype),
            Weight(hidden_size, hidden_size, dtype=dtype),
            Bias(hidden_size, dtype=dtype))



