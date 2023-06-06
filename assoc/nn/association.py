import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from assoc.types import *
from assoc.nn.basic import *


# TODO: Make attention to de unit of length between cities. It is KM now.
class MapGC(torch.nn.Module):
    def __init__(self,
                 map_units: int,
                 in_channels: int,
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 200.,
                 out_chennels: int = 1,
                 k: int = 2,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.map_units = map_units
        self.in_channels = in_channels
        self.out_channels = out_chennels
        self.k = k
        self.feature_last = feature_last
        self.dtype = dtype
        within_threshold = (dist_mat > 0) & (dist_mat < dist_threshold)
        self.edge_mat = torch.where(within_threshold, dist_mat, 0)
        self.conv = ChebConv(self.in_channels, self.out_channels, K=k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.dim()
        assert (shape[-2:] == (self.map_units, self.in_channels)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.map_units}, {self.in_channels}), but received {shape}'
        )

        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        x_gcn = x.reshape(batch_size * self.map_units, self.in_channels)
        x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_mat))
        x_gcn = x_gcn.reshape(batch_size, self.map_units, self.out_channels)

        y = torch.cat((x, x_gcn), dim=-1)
        return y if is_batched else y.squeeze(0)