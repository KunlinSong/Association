import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from assoc.types import *
from assoc.nn.basic import *

__all__ = ['AssociationModule', 'GAT', 'GC', 'GCN', 'INA']


class GC(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 add_channels: Optional[int],
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: Union[int, float] = 2e5,
                 k: int = 2,
                 eps: float = 1e-5,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.add_channels = add_channels if add_channels is not None else 1
        self.k = k
        self.feature_last = feature_last
        self.dtype = dtype
        self.edge_mat = ((dist_mat > 0) & (dist_mat < dist_threshold)).to(
            torch.int64)
        # within_threshold = (dist_mat > 0) & (dist_mat < dist_threshold)
        # self.edge_mat = within_threshold / (dist_mat + eps)
        self.conv = ChebConv(in_channels, add_channels, K=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.edge_mat = self.edge_mat.to(x.device)
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
        x_gcn = x_gcn.reshape(batch_size, self.nodes, self.add_channels)

        y = torch.cat((x, x_gcn), dim=-1)
        return y if is_batched else y.squeeze(0)


class GCN(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 out_channels: Optional[int],
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 2e5,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.out_channels = (out_channels
                             if out_channels is not None else in_channels)
        self.feature_last = feature_last
        self.dtype = dtype
        self.adj_mat = (
            (dist_mat > 0) &
            (dist_mat < dist_threshold)).to(dtype if dtype else torch.float)
        self.adj_mat.fill_diagonal_(1)
        self.neighbors = self.adj_mat.sum(dim=-1, keepdims=True)
        self.projection = torch.nn.Linear(in_features=self.in_channels,
                                          out_features=self.out_channels,
                                          dtype=dtype)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x: torch.Tensor):
        self.adj_mat = self.adj_mat.to(x.device)
        self.neighbors = self.neighbors.to(x.device)
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
                 hidden_channels: Optional[int],
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 2e5,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 alpha: float = 0.01):
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels else 32
        self.feature_last = feature_last
        self.dtype = dtype
        self.adj_mat = (
            (dist_mat > 0) &
            (dist_mat < dist_threshold)).to(dtype if dtype else torch.float)
        self.adj_mat.fill_diagonal_(1)

        self.projection = torch.nn.Linear(in_features=in_channels,
                                          out_features=hidden_channels,
                                          dtype=dtype)
        self.a = torch.nn.Linear(in_features=hidden_channels * 2,
                                 out_features=1,
                                 bias=False,
                                 dtype=dtype)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x: torch.Tensor):
        self.adj_mat = self.adj_mat.to(x.device)
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
                                               self.hidden_channels),
            x.repeat(1, self.nodes, 1)
        ],
                             dim=-1)
        x_concat = x_concat.reshape(-1, self.nodes, self.nodes,
                                    self.hidden_channels * 2)
        x_activated = self.leakyrelu(self.a(x_concat)).reshape(
            -1, self.nodes, self.nodes)
        attention_mat = torch.where(self.adj_mat > 0, x_activated, -1e15)
        attention_mat = F.softmax(attention_mat, dim=-1)
        y = torch.einsum('bij, bjc -> bic', attention_mat, x)
        return y if is_batched else y.squeeze(0)


class INA(torch.nn.Module):

    def __init__(self,
                 nodes: int,
                 in_channels: int,
                 hidden_channels: Optional[int],
                 location_mat: torch.Tensor,
                 time_features_index: Sequence[int],
                 *,
                 feature_last: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 alpha: float = 0.01):
        super().__init__()
        self.nodes = nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels else 32
        self.feature_last = feature_last
        self.dtype = dtype
        self.location_mat = location_mat
        self.time_features_index = time_features_index

        self.location_cat_mat = torch.cat([
            location_mat.repeat(1, nodes).reshape(nodes**2, -1),
            location_mat.repeat(nodes, 1)
        ],
                                          dim=-1)

        self.a = torch.nn.Linear(in_features=location_mat.shape[-1] * 2,
                                 out_features=hidden_channels,
                                 dtype=dtype)
        self.dense = torch.nn.Linear(in_features=hidden_channels,
                                     out_features=1,
                                     dtype=dtype)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        self.time_mask_transformer = torch.nn.Linear(len(time_features_index),
                                                     hidden_channels,
                                                     dtype=dtype)
        torch.nn.init.xavier_uniform_(self.a.weight)
        torch.nn.init.xavier_uniform_(self.time_mask_transformer.weight)

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
        time_mask = F.sigmoid(time_mask)
        time_mask = time_mask.unsqueeze(1)

        self.location_cat_mat = self.location_cat_mat.to(x.device)
        assoc_mat = self.a(self.location_cat_mat)
        assoc_mat = assoc_mat.unsqueeze(0)
        assoc_mat = assoc_mat * time_mask
        assoc_mat = self.dense(assoc_mat)
        assoc_mat = self.leakyrelu(assoc_mat)
        assoc_mat = assoc_mat.reshape(-1, self.nodes, self.nodes)
        return F.softmax(assoc_mat, dim=-1)


AssociationModule = Union[GC, GCN, GAT, INA]