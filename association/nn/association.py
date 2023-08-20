import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse

from association.types import *
from association.nn.basic import *

__all__ = ['AssociationModule', 'GAT', 'GC', 'GCN', 'INA']


class GC(torch.nn.Module):
    def __init__(self, 
                 num_nodes: int, 
                 num_in_channels: int, 
                 num_add_channels: Optional[int], 
                 dist_mat: Union[torch.Tensor, np.ndarray],
                 *,
                 dist_threshold: Union[int, float] = 2e5,
                 k: int = 2,
                 eps: float = 1e-5,
                 dtype: str = 'float32') -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_add_channels = num_add_channels
        dtype = getattr(torch, dtype)
        num_add_channels = 1 if num_add_channels is None else num_add_channels
        mask = dist_mat <= dist_threshold
        dist_mat = dist_mat * mask
        self.edge_idx, edge_w = dense_to_sparse(torch.tensor(1 / (dist_mat + eps)))
        self.edge_w = edge_w.to(dtype)
        self.conv = ChebConv(num_in_channels, num_add_channels, K=k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.edge_idx = self.edge_idx.to(x.device)
        self.edge_w = self.edge_w.to(x.device)
        y = self.conv(x, self.edge_idx, self.edge_w)
        y = F.sigmoid(y)
        y = y.reshape(-1, self.num_nodes, self.num_add_channels)
        y = torch.cat((x, y), dim=-1)
        return y


class GCN(torch.nn.Module):

    def __init__(self,
                 num_in_channels: int,
                 num_out_channels: Optional[int],
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 2e5,
                 dtype: str = 'float32'):
        super().__init__()
        dtype = getattr(torch, dtype)
        num_out_channels = num_in_channels if num_out_channels is None else num_out_channels
        adj_mat = (dist_mat < dist_threshold)
        adj_mat = adj_mat.to(dtype)
        adj_mat.fill_diagonal_(1)
        neighbors = adj_mat.sum(dim=-1, keepdims=True)
        self.adj_mat = adj_mat / neighbors
        self.projection = torch.nn.Linear(in_features=num_in_channels,
                                          out_features=num_out_channels,
                                          dtype=dtype)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x: torch.Tensor):
        self.adj_mat = self.adj_mat.to(x.device)
        x = self.projection(x)
        node_feats = torch.einsum('ij, ...jc -> ...ic', self.adj_mat, x)
        return node_feats


class GAT(torch.nn.Module):

    def __init__(self,
                 num_nodes: int,
                 num_in_channels: int,
                 num_hidden_channels: Optional[int],
                 dist_mat: torch.Tensor,
                 *,
                 dist_threshold: float = 2e5,
                 alpha: float = 0.01,
                 dtype: str = 'float32'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_hidden_channels = 32 if num_hidden_channels is None else num_hidden_channels
        dtype = getattr(torch, dtype)
        adj_mat = (dist_mat < dist_threshold)
        self.adj_mat = adj_mat.to(dtype)
        self.adj_mat.fill_diagonal_(1)

        self.projection = torch.nn.Linear(in_features=num_in_channels,
                                          out_features=num_hidden_channels,
                                          dtype=dtype)
        self.a = torch.nn.Linear(in_features=num_hidden_channels * 2,
                                 out_features=1,
                                 bias=False,
                                 dtype=dtype)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x: torch.Tensor):
        self.adj_mat = self.adj_mat.to(x.device)
        x = self.projection(x)
        is_batched = x.dim() == 3
        x_concat = torch.cat([
            torch.repeat_interleave(x, self.num_nodes, dim=-1).reshape(-1, self.num_nodes**2,
                                               self.num_hidden_channels),
            torch.repeat_interleave(x, self.num_nodes, dim=-2).reshape(-1, self.num_nodes**2,
                                                                       self.num_hidden_channels)
        ],
                             dim=-1)
        x_concat = x_concat.reshape(-1, self.num_nodes, self.num_nodes,
                                    self.num_hidden_channels * 2)
        x_activated = self.leakyrelu(self.a(x_concat)).reshape(
            -1, self.num_nodes, self.num_nodes)
        attention_mat = torch.where(self.adj_mat > 0, x_activated, -1e15)
        attention_mat = F.softmax(attention_mat, dim=-1)
        y = torch.einsum('...ij, ...jc -> ...ic', attention_mat, x)
        return y if is_batched else y.squeeze(0)


class INA(torch.nn.Module):

    def __init__(self,
                 num_nodes: int,
                 num_in_channels: int,
                 num_hidden_channels: Optional[int],
                 *,
                 alpha: float = 0.01,
                 dtype: str = 'float32'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_hidden_channels = 32 if num_hidden_channels is None else num_hidden_channels
        dtype = getattr(torch, dtype)

        # self.w = Weight((num_in_channels,), dtype=dtype)
        self.w = torch.nn.Parameter(torch.randn((num_in_channels), dtype=dtype))
        self.b = Bias((num_in_channels,), dtype=dtype)

        self.projection = torch.nn.Linear(in_features=num_in_channels,
                                          out_features=num_hidden_channels,
                                          dtype=dtype)
        self.node_eval = torch.nn.Linear(in_features=num_hidden_channels,
                                         out_features=1,
                                         bias=False,
                                         dtype=dtype)
        self.a = torch.nn.Linear(in_features=2,
                                 out_features=1,
                                 bias=False,
                                 dtype=dtype)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.xavier_uniform_(self.a.weight)


    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        is_batched = x.dim() == 3

        y = x * self.w + self.b
        y = self.projection(y)
        y = self.node_eval(y).reshape(-1, self.num_nodes, 1)

        y_concat = torch.cat([
            torch.repeat_interleave(y, self.num_nodes, dim=-1).reshape(-1, self.num_nodes**2, 1),
            torch.repeat_interleave(y, self.num_nodes, dim=-2).reshape(-1, self.num_nodes**2, 1)
        ], dim=-1)
        y_concat = y_concat.reshape(-1, self.num_nodes, self.num_nodes, 2)
        x_activated = self.leakyrelu(self.a(y_concat)).reshape(
            -1, self.num_nodes, self.num_nodes)
        attention_mat = F.softmax(x_activated, dim=-1)
        return attention_mat if is_batched else attention_mat.squeeze(0)