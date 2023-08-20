import numpy as np
import torch
import torch.nn.functional as F

import association.nn.association as association
import association.nn.rnncell as rnncell
from association.types import *


class Model(torch.nn.Module):

    def __init__(self,
                 association_mode: Optional[str],
                 association_param: Optional[str],
                 rnn_mode: str,
                 rnn_hidden: Optional[int],
                 in_features: int,
                 out_features: int,
                 input_time_steps: int,
                 num_nodes: int,
                 distance_matrix: np.ndarray,
                 adjacency_threshold: Optional[float] = 2e5,
                 dtype: str = 'float32') -> None:
        super().__init__()
        self.association_mode = association_mode
        self.association_param = association_param
        self.rnn_mode = rnn_mode
        self.rnn_hidden = rnn_hidden
        self.in_features = in_features
        self.out_features = out_features
        self.input_time_steps = input_time_steps
        self.num_nodes = num_nodes
        self.distance_matrix = torch.tensor(distance_matrix)
        self.adjacency_threshold = adjacency_threshold
        self.dtype = dtype
        self._init_model()

    def _init_model(self):
        self.batch_norm = torch.nn.BatchNorm2d(self.in_features)

        match self.association_mode:
            case 'GC':
                self.assoc_layer = association.GC(
                    self.num_nodes,
                    self.in_features,
                    self.association_param,
                    self.distance_matrix,
                    dist_threshold=self.adjacency_threshold,
                    dtype=self.dtype)
                self.rnn_in_features = self.association_param + self.in_features
            case 'GCN':
                self.assoc_layer = association.GCN(
                    self.in_features,
                    self.association_param,
                    self.distance_matrix,
                    dist_threshold=self.adjacency_threshold,
                    dtype=self.dtype)
                self.rnn_in_features = self.association_param
            case 'GAT':
                self.assoc_layer = association.GAT(
                    self.num_nodes,
                    self.in_features,
                    self.association_param,
                    self.distance_matrix,
                    dist_threshold=self.adjacency_threshold,
                    dtype=self.dtype)
                self.rnn_in_features = self.association_param
            case 'INA':
                self.assoc_layer = association.INA(
                    self.num_nodes,
                    self.in_features,
                    self.association_param,
                    dtype=self.dtype
                )
                self.rnn_in_features = self.in_features
            case _:
                self.assoc_layer = None
                self.rnn_in_features = self.in_features

        match self.rnn_mode:
            case 'LSTM':
                self.rnn_layer = rnncell.GraphLSTMCell(self.num_nodes,
                                                    self.rnn_in_features,
                                                    self.rnn_hidden)
            case 'GRU':
                self.rnn_layer = rnncell.GraphGRUCell(self.num_nodes,
                                                    self.rnn_in_features,
                                                    self.rnn_hidden)
            case 'RNN':
                self.rnn_layer = rnncell.GraphRNNCell(self.num_nodes,
                                                    self.rnn_in_features,
                                                    self.rnn_hidden)
            case _:
                raise ValueError(f'Unknown rnn mode: {self.rnn_mode}')

        self.dense_1 = torch.nn.Linear(self.rnn_hidden, self.rnn_hidden)
        self.dense_2 = torch.nn.Linear(self.rnn_hidden, self.out_features)

    def forward(self, x: torch.Tensor, output_mat: bool=False):
        x = x.transpose(1, -1)
        x = self.batch_norm(x)
        x = x.transpose(1, -1)

        x = x.transpose(0, 1)
        result = []
        state = None
        for i in range(self.input_time_steps):
            y = x[i]

            if self.assoc_layer is not None:
                if self.association_mode == 'INA':
                    assoc_mat = self.assoc_layer(y)
                    if output_mat:
                        return assoc_mat
                else:
                    y = self.assoc_layer(y)

            state = self.rnn_layer(y, state=state)
            y = state[0].clone()
            if self.association_mode == 'INA':
                y = torch.matmul(assoc_mat, y)
            result.append(y)
        result = torch.stack(result)
        result = result.transpose(0, 1)
        result = self.dense_1(result)
        result = F.relu(result)
        result = self.dense_2(result)
        return result