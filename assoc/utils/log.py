import os
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard as tensorboard

from assoc.types import *
from assoc.utils.config import ConfigHub


class StatisticalMeasures:

    def __init__(self, true_val: np.ndarray, pred_val: np.ndarray) -> None:
        self.true_val = true_val
        self.pred_val = pred_val

    @property
    def mae(self) -> float:
        return np.mean(np.abs(self.true_val - self.pred_val))

    @property
    def mse(self) -> float:
        return np.mean((self.true_val - self.pred_val)**2)

    @property
    def rmse(self) -> float:
        return np.sqrt(np.mean((self.true_val - self.pred_val)**2))

    @property
    def mape(self) -> float:
        return np.mean(
            np.abs(self.true_val - self.pred_val) / self.true_val) * 100

    @property
    def r2(self) -> float:
        return (1 - np.sum((self.true_val - self.pred_val)**2) / np.sum(
            (self.true_val - np.mean(self.true_val))**2))

    @property
    def pearson(self) -> float:
        return np.corrcoef(self.true_val, self.pred_val)[0, 1]

    @property
    def ia(self) -> float:
        return (1 - np.sum((self.true_val - self.pred_val)**2) / np.sum(
            (np.abs(self.pred_val - np.mean(self.pred_val)) +
             np.abs(self.true_val - np.mean(self.true_val)))**2))


class Plot:

    def __init__(self,
                 dirname: str,
                 true_val: np.ndarray,
                 pred_val: np.ndarray,
                 default_min_val: Union[float, int] = 0,
                 default_max_val: Optional[Union[float, int]] = None) -> None:
        self.dirname = dirname
        self.true_val = true_val
        self.pred_val = pred_val
        self.min_val = min(np.min(self.true_val), np.min(self.pred_val),
                           default_min_val)
        if default_max_val is None:
            self.max_val = max(np.max(self.true_val), np.max(self.pred_val))
        else:
            self.max_val = max(np.max(self.true_val), np.max(self.pred_val),
                               default_max_val)

    def save_hexbin(self) -> None:
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        path = os.path.join(self.dirname, 'hexbin.png')
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        ax.hexbin(self.true_val, self.pred_val, cmap='jet')
        cbar = fig.colorbar(ax.get_children()[0], ax=ax)
        cbar.set_label('Counts')
        ax.plot((self.min_val, self.min_val), (self.max_val, self.max_val),
                color="white",
                label="Perfect Prediction")
        ax.set_xlim(self.min_val, self.max_val)
        ax.set_ylim(self.min_val, self.max_val)
        ax.set_aspect('equal')
        ax.set_xlabel(f'Observed ({chr(0x03BC)}g/m{chr(0x00B3)})')
        ax.set_ylabel(f'Predicted ({chr(0x03BC)}g/m{chr(0x00B3)})')
        fig.savefig(path, dpi=300)

    def save_plot(self) -> None:
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        path = os.path.join(self.dirname, 'plot.png')
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        ax.plot(self.true_val, label='Observed', c='blue')
        ax.plot(self.pred_val, label='Predicted', c='orange')
        ax.legend(loc='lower left', ncol=1)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'Value ({chr(0x03BC)}g/m{chr(0x00B3)})')
        fig.savefig(path, dpi=300)


class ConfigLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
            self.config_hub = None
        else:
            self.config_hub = ConfigHub(self.dirname)

    def load_config_hub(self, config_hub: ConfigHub) -> None:
        if config_hub is None:
            raise ValueError('ConfigHub cannot be None')
        self.config_hub = config_hub
        self.config_hub.save(self.dirname)


class ModelLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.latest_path = os.path.join(self.dirname, 'latest.pth')
        self.best_path = os.path.join(self.dirname, 'best.pth')

    @property
    def best_state_dict(self) -> Any:
        if os.path.exists(self.best_path):
            return torch.load(self.best_path)
        else:
            raise FileNotFoundError(f'Best model not found in {self.dirname}')

    @property
    def latest_state_dict(self) -> Any:
        if os.path.exists(self.latest_path):
            return torch.load(self.latest_path)
        else:
            raise FileNotFoundError(
                f'Latest model not found in {self.dirname}')

    def save_best_state_dict(self, state_dict: Any) -> None:
        torch.save(state_dict, self.best_path)

    def save_latest_state_dict(self, state_dict: Any) -> None:
        torch.save(state_dict, self.latest_path)


class TrainingLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.log = os.path.join(self.dirname, 'training.csv')

        if os.path.exists(self.log):
            self.df = pd.read_csv(self.log)
        else:
            self.df = pd.DataFrame(
                columns=['epoch', 'training_loss', 'validation_loss'])

    def append(self, epoch: int, training_loss: float,
               validation_loss: float) -> None:
        if not self.df['epoch']:
            self.df.iloc[0] = [epoch, training_loss, validation_loss]
        else:
            if epoch != (self.latest_epoch + 1):
                raise ValueError(f'Epoch {epoch} is not the next epoch')
            else:
                self.df = pd.concat([
                    self.df,
                    pd.DataFrame(
                        [[epoch, training_loss, validation_loss]],
                        columns=['epoch', 'training_loss', 'validation_loss'])
                ])
        self.df.to_csv(self.log, index=False, header=True)

    @property
    def latest_epoch(self) -> int:
        return self.df['epoch'].iloc[-1]

    @property
    def best_epoch_info(self) -> tuple[int, float, float]:
        if self.df.empty:
            return (0, float('inf'), float('inf'))
        else:
            best_idx = self.df['validation_loss'].idxmin()
            return (self.df['epoch'].iloc[best_idx],
                    self.df['training_loss'].iloc[best_idx],
                    self.df['validation_loss'].iloc[best_idx])


class NodeTestLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        self.path = os.path.join(self.dirname, 'test_log.csv')
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=['true_val', 'pred_val'])

    def append(self, true_val: float, pred_val: float) -> None:
        self.df = pd.concat([
            self.df,
            pd.DataFrame([[true_val, pred_val]],
                         columns=['true_val', 'pred_val'])
        ])
        self.df.to_csv(self.path, index=False)

    @property
    def true_val(self) -> np.ndarray:
        return self.df['true_val'].values

    @property
    def pred_val(self) -> np.ndarray:
        return self.df['pred_val'].values

    @property
    def statistical_measures(self) -> StatisticalMeasures:
        return StatisticalMeasures(self.true_val, self.pred_val)

    def plot(self,
             default_min_val: Union[float, int] = 0,
             default_max_val: Optional[Union[float, int]] = None) -> Plot:
        return Plot(self.dirname, self.true_val, self.pred_val,
                    default_min_val, default_max_val)


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

    def statistical_measures(
        self,
        nodes: Optional[Union[str,
                              Sequence[str]]] = None) -> StatisticalMeasures:
        return StatisticalMeasures(self.true_val(nodes), self.pred_val(nodes))

    def plot(self,
             nodes: Optional[Union[str, Sequence[str]]] = None,
             default_min_val: Union[float, int] = 0,
             default_max_val: Optional[Union[float, int]] = None) -> Plot:
        return Plot(self.dirname, self.true_val(nodes), self.pred_val(nodes),
                    default_min_val, default_max_val)


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
            target: os.path.join(self.dirname, target)
            for target in self.targets
        }

    def __getitem__(self, target: str) -> TargetTestLog:
        return TargetTestLog(self.target_logs[target], self.nodes)

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

    def statistical_measures(
        self,
        targets: Optional[Union[str, Sequence[str]]] = None,
        nodes: Optional[Union[str,
                              Sequence[str]]] = None) -> StatisticalMeasures:
        return StatisticalMeasures(self.true_val(targets, nodes),
                                   self.pred_val(targets, nodes))

    def plot(self,
             targets: Optional[Union[str, Sequence[str]]] = None,
             nodes: Optional[Union[str, Sequence[str]]] = None,
             default_min_val: Union[float, int] = 0,
             default_max_val: Optional[Union[float, int]] = None) -> Plot:
        return Plot(self.dirname, self.true_val(targets, nodes),
                    self.pred_val(targets, nodes), default_min_val,
                    default_max_val)


class TensorboardLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        self.writer = tensorboard.SummaryWriter(self.dirname)

    def add_graph(self, model: torch.nn.Module,
                  input_shape: Sequence[int]) -> None:
        fake_input = torch.rand(*input_shape,
                                dtype=model.dtype,
                                device=model.device)
        self.writer.add_graph(model, fake_input)

    def append_loss(self, epoch: int, training_loss: float,
                    validation_loss: float) -> None:
        self.writer.add_scalar('loss/training', training_loss, epoch)
        self.writer.add_scalar('loss/validation', validation_loss, epoch)

    def append_test(self, idx: int, target: str, node: str, true_val: float,
                    pred_val: float) -> None:
        self.writer.add_scalar(f'test/{target}/{node}/true_val', true_val, idx)
        self.writer.add_scalar(f'test/{target}/{node}/pred_val', pred_val, idx)


class LogHub:

    CONFIG_FOLDER = 'config'
    MODEL_FOLDER = 'model'
    TRAINING_FOLDER = 'training'
    TEST_FOLDER = 'test'
    TENSORBOARD_FOLDER = 'tensorboard'
    LOG_FOLDERS = [
        CONFIG_FOLDER, MODEL_FOLDER, TRAINING_FOLDER, TEST_FOLDER,
        TENSORBOARD_FOLDER
    ]

    def __init__(self,
                 dirname: str,
                 config_hub: Optional[ConfigHub] = None) -> None:

        self.dirname = dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            config_hub = ConfigHub.from_log_hub(self.dirname)
        self.config_log = ConfigLog(self._get_log_dir(LogHub.CONFIG_FOLDER))
        self.config_log.load_config_hub(config_hub)
        self.test_log = TestLog(
            self._get_log_dir(LogHub.TEST_FOLDER),
            self.config_log.config_hub.data_config.all_cities,
            self.config_log.config_hub.data_config.targets)
        self.model_log = ModelLog(self._get_log_dir(LogHub.MODEL_FOLDER))
        self.training_log = TrainingLog(
            self._get_log_dir(LogHub.TRAINING_FOLDER))
        self.tensorboard_log = TensorboardLog(
            self._get_log_dir(LogHub.TENSORBOARD_FOLDER))

    def _get_log_dir(self, log_type: str) -> str:
        return os.path.join(self.dirname, log_type)