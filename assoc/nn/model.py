import numpy as np
import torch

import assoc.nn.association as association
import assoc.nn.rnncell as rnncell
from assoc.types import *


class Model(torch.nn.Module):

    def __init__(self,
                 assoc_mode: Optional[str],
                 assoc_channels: Optional[str],
                 rnn_mode: str,
                 rnn_hidden: Optional[int],
                 in_features: int,
                 out_features: int,
                 input_time_steps: int,
                 nodes: int,
                 dist_mat: np.ndarray,
                 time_features_index: Sequence[int],
                 location_mat: np.ndarray,
                 adjacency_threshold: Optional[float] = 2e5) -> None:
        super().__init__()
        self.assoc_mode = assoc_mode
        self.assoc_channels = assoc_channels
        self.rnn_mode = rnn_mode
        self.rnn_hidden = rnn_hidden
        self.in_features = in_features
        self.out_features = out_features
        self.input_time_steps = input_time_steps
        self.nodes = nodes
        self.dist_mat = torch.tensor(dist_mat)
        self.time_features_index = time_features_index
        self.location_mat = torch.tensor(location_mat)
        self.adjacency_threshold = adjacency_threshold
        self._init_model()

    def _init_model(self):
        if self.assoc_mode == 'GC':
            if self.assoc_channels is None:
                self.assoc_channels = 1
            self.assoc_layer = association.GC(
                self.nodes,
                self.in_features,
                self.assoc_channels,
                self.dist_mat,
                dist_threshold=self.adjacency_threshold)
            self.rnn_in_features = self.assoc_channels + self.in_features
        elif self.assoc_mode == 'GCN':
            if self.assoc_channels is None:
                self.assoc_channels = self.in_features
            self.assoc_layer = association.GCN(
                self.nodes,
                self.in_features,
                self.assoc_channels,
                self.dist_mat,
                dist_threshold=self.adjacency_threshold)
            self.rnn_in_features = self.assoc_channels
        elif self.assoc_mode == 'GAT':
            if self.assoc_channels is None:
                self.assoc_channels = 32
            self.assoc_layer = association.GAT(
                self.nodes,
                self.in_features,
                self.assoc_channels,
                self.dist_mat,
                dist_threshold=self.adjacency_threshold)
            self.rnn_in_features = self.assoc_channels
        elif self.assoc_mode == 'INA':
            if self.assoc_channels is None:
                self.assoc_channels = 32
            self.assoc_layer = association.INA(
                self.nodes,
                self.in_features,
                self.assoc_channels,
                self.location_mat,
                self.time_features_index
            )
            self.rnn_in_features = self.in_features
        else:
            self.assoc_layer = None
            self.rnn_in_features = self.in_features

        self.batch_norm = torch.nn.BatchNorm2d(self.in_features)

        if self.rnn_mode == 'LSTM':
            self.rnn_layer = rnncell.GraphLSTMCell(self.nodes,
                                                   self.rnn_in_features,
                                                   self.rnn_hidden)
        elif self.rnn_mode == 'GRU':
            self.rnn_layer = rnncell.GraphGRUCell(self.nodes,
                                                  self.rnn_in_features,
                                                  self.rnn_hidden)
        elif self.rnn_mode == 'RNN':
            self.rnn_layer = rnncell.GraphRNNCell(self.nodes,
                                                  self.rnn_in_features,
                                                  self.rnn_hidden)
        else:
            raise ValueError(f'Unknown rnn mode: {self.rnn_mode}')

        self.dense = torch.nn.Linear(self.rnn_hidden, self.out_features)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, -1)
        x = self.batch_norm(x)
        x = x.transpose(1, -1)

        x = x.transpose(0, 1)
        result = []
        state = None
        for i in range(self.input_time_steps):
            y = x[i]

            if self.assoc_layer is not None:
                if self.assoc_mode == 'INA':
                    assoc_mat = self.assoc_layer(y)
                else:
                    y = self.assoc_layer(y)

            state = self.rnn_layer(y, state=state)
            y = state[0]
            if self.assoc_mode == 'INA':
                y = torch.matmul(assoc_mat, y)
            result.append(y)
        result = torch.stack(result)
        result = result.transpose(0, 1)
        result = self.dense(result)
        return result