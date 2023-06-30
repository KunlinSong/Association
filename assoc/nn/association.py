import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from assoc.types import *
from assoc.nn.basic import *

__all__ = ['GC']


# TODO: Make attention to the unit of length between cities. It is km now.
class GC(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: Union[int, float] = 200,
                 out_channels: int = 1,
                 k: int = 2,
                 eps: float = 1e-5,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.feature_last = feature_last
        self.dtype = dtype
        within_threshold = (dist_mat > 0) & (dist_mat < dist_threshold)
        self.edge_mat = within_threshold / (dist_mat + eps)
        self.conv = ChebConv(self.in_channels, self.out_channels, K=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.shape
        assert (shape[-2:] == (self.nodes, self.in_channels)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.in_channels}), but received {shape}')

        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        x_gcn = x.reshape(batch_size * self.nodes, self.in_channels)
        x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_mat))
        x_gcn = x_gcn.reshape(batch_size, self.nodes, self.out_channels)

        y = torch.cat((x, x_gcn), dim=-1)
        return y if is_batched else y.squeeze(0)


class GCN(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 200,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.feature_last = feature_last
        self.dtype = dtype
        self.adj_mat = (
            (dist_mat > 0) &
            (dist_mat < dist_threshold)).to(dtype if dtype else torch.float)
        self.adj_mat.fill_diagonal_(1)
        self.neighbors = self.adj_mat.sum(dim=-1, keepdims=True)
        self.projection = GraphDense(nodes=nodes,
                                     in_features=in_channels,
                                     out_features=in_channels,
                                     feature_last=True,
                                     dtype=dtype)

    def forward(self, x: torch.Tensor):
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.shape
        assert (shape[-2:] == (self.nodes, self.in_channels)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.in_channels}), but received {shape}')

        x = self.projection(x)
        node_feats = torch.einsum('ij, bjc -> bic', self.adj_mat, x)
        node_feats = node_feats / self.neighbors
        return node_feats


class GAT(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 200,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 alpha: float = 0.01):
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.feature_last = feature_last
        self.dtype = dtype
        self.adj_mat = (
            (dist_mat > 0) &
            (dist_mat < dist_threshold)).to(dtype if dtype else torch.float)
        self.adj_mat.fill_diagonal_(1)

        self.projection = GraphDense(nodes=nodes,
                                     in_features=in_channels,
                                     out_features=in_channels,
                                     feature_last=True,
                                     dtype=dtype)
        self.a = GraphDense(nodes=nodes,
                            in_features=self.in_channels * 2,
                            out_features=1,
                            feature_last=True,
                            bias=False,
                            dtype=dtype)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor):
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.shape
        assert (shape[-2:] == (self.nodes, self.in_channels)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.in_channels}), but received {shape}')

        x = self.projection(x)
        is_batched = dim == 3
        x = x if is_batched else x.unsqueeze(0)

        x_concat = torch.cat([
            x.repeat(1, 1, self.nodes).reshape(-1, self.nodes**2,
                                               self.in_channels),
            x.repeat(1, self.nodes, 1)
        ],
                             dim=-1)
        x_concat = x_concat.reshape(-1, self.nodes, self.nodes,
                                    self.in_channels * 2)
        x_activated = self.leakyrelu(self.a(x_concat)).reshape(
            -1, self.nodes, self.nodes)
        attention_mat = torch.where(self.adj_mat > 0, x_activated, -1e15)
        attention_mat = F.softmax(attention_mat, dim=-1)
        y = torch.einsum('bij, bjc -> bic', attention_mat, x)
        return y if is_batched else y.squeeze(0)


class ICA(torch.nn.Module):
    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 location_mat: torch.Tensor,
                 time_features_index: Sequence[int],
                 *,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 alpha: float = 0.01):
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.feature_last = feature_last
        self.dtype = dtype
        self.location_mat = location_mat
        self.time_features_index = time_features_index

        self.location_cat_mat = torch.cat([
            location_mat.repeat(1, nodes).reshape(nodes**2,
                                               -1),
            location_mat.repeat(nodes, 1)
        ],
                             dim=-1)
        
        self.a = GraphDense(nodes=nodes,
                            in_features=len(time_features_index) * 2,
                            out_features=1,
                            feature_last=True,
                            dtype=dtype)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        self.time_mask_transformer = torch.nn.Linear(len(time_features_index), nodes ** 2, dtype=dtype)

    def forward(self, x: torch.Tensor):
        dim = x.dim()
        assert dim in (2, 3), (
            f'{self.__class__.__name__}: Expected x to be 2-D or 3-D, but '
            f'received {dim}-D')
        if not self.feature_last:
            x = x.transpose(-1, -2)
        shape = x.shape
        assert (shape[-2:] == (self.nodes, self.in_channels)), (
            f'{self.__class__.__name__}: Expected x to be torch.Tensor of shape'
            f' (*, {self.nodes}, {self.in_channels}), but received {shape}')
        is_batched = dim == 3
        x = x if is_batched else x.unsqueeze(0)
        x_time = x[:, 0, self.time_features_index]
        time_mask = self.time_mask_transformer(x_time)
        time_mask = F.sigmoid(time_mask).reshape(-1, self.nodes, self.nodes)

        assoc_mat = self.a(self.location_cat_mat)
        assoc_mat = assoc_mat.reshape(-1, self.nodes, self.nodes)
        assoc_mat = self.leakyrelu(assoc_mat)
        assoc_mat = self.leakyrelu(assoc_mat * time_mask)
        return F.softmax(assoc_mat, dim=-1)


AssociationModule = Union[GC, GCN, GAT]