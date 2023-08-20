import os

import numpy as np
import pandas as pd

from association.types import *


TRUE_COL = 'true_val'
PRED_COL = 'pred_val'


class NodeTestLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.path = os.path.join(self.dirname, 'test_log.csv')
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=[TRUE_COL, PRED_COL])

    def append(self, true_val: float, pred_val: float) -> None:
        self.df = pd.concat([
            self.df,
            pd.DataFrame([[true_val, pred_val]],
                         columns=[TRUE_COL, PRED_COL])
        ])
        self.df.to_csv(self.path, index=False)

    @property
    def true_val(self) -> np.ndarray:
        return self.df[TRUE_COL].values

    @property
    def pred_val(self) -> np.ndarray:
        return self.df[PRED_COL].values


class TargetTestLog:

    def __init__(self, dirname: str, nodes: Union[str, Sequence[str]]) -> None:
        self.dirname = dirname

        if isinstance(nodes, str):
            self.nodes = [nodes]
        else:
            self.nodes = list(nodes)

        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        self.node_logs = {
            node: NodeTestLog(os.path.join(self.dirname, node))
            for node in self.nodes
        }

    def __getitem__(self, node: str) -> NodeTestLog:
        return self.node_logs[node]

    def true_val(
            self,
            nodes: Optional[Union[str, Sequence[str]]] = None) -> np.ndarray:
        if nodes is None:
            nodes = self.nodes
        elif isinstance(nodes, str):
            nodes = [nodes]
        else:
            nodes = list(nodes)
        return np.concatenate(
            [self.node_logs[node].true_val for node in nodes])

    def pred_val(
            self,
            nodes: Optional[Union[str, Sequence[str]]] = None) -> np.ndarray:
        if nodes is None:
            nodes = self.nodes
        elif isinstance(nodes, str):
            nodes = [nodes]
        else:
            nodes = list(nodes)
        return np.concatenate(
            [self.node_logs[node].pred_val for node in nodes])


class TestLog:

    def __init__(self, dirname: str, nodes: Union[str, Sequence[str]],
                 targets: Union[str, Sequence[str]]) -> None:
        self.dirname = dirname
        self.prediction = os.path.join(dirname, 'prediction')

        if isinstance(nodes, str):
            self.nodes = [nodes]
        else:
            self.nodes = nodes

        if isinstance(targets, str):
            self.targets = [targets]
        else:
            self.targets = targets

        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.target_logs = {
            target: TargetTestLog(os.path.join(self.dirname, target), self.nodes)
            for target in self.targets
        }

    def __getitem__(self, target: str) -> TargetTestLog:
        return self.target_logs[target]

    def true_val(
            self,
            targets: Optional[Union[str, Sequence[str]]] = None,
            nodes: Optional[Union[str, Sequence[str]]] = None) -> np.ndarray:
        if targets is None:
            targets = self.targets
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        return np.concatenate(
            [self.target_logs[target].true_val(nodes) for target in targets])

    def pred_val(
            self,
            targets: Optional[Union[str, Sequence[str]]] = None,
            nodes: Optional[Union[str, Sequence[str]]] = None) -> np.ndarray:
        if targets is None:
            targets = self.targets
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        return np.concatenate(
            [self.target_logs[target].pred_val(nodes) for target in targets])