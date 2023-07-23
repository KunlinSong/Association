import datetime
import os
import re
from typing import overload

import numpy as np
import yaml

import assoc
from assoc.types import *


class Config:

    def __init__(self, path: str) -> None:
        self._path = path
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)

    @property
    def path(self) -> str:
        return self._path

    @property
    def dirname(self) -> str:
        return os.path.dirname(self.path)

    @overload
    def save(self, path: str) -> None:
        ...

    @overload
    def save(self, dirname: str, filename: str) -> None:
        ...

    def save(self, **kwargs) -> None:
        path = (kwargs.get('path') if ('path' in kwargs) else os.path.join(
            kwargs.get('dirname'), kwargs.get('filename')))
        if os.path.exists(path):
            raise FileExistsError(f'File {path} already exists.')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as file:
            yaml.dump(self._config, file, indent=4, sort_keys=False)

    def __eq__(self, other: 'Config') -> bool:
        return self._config == other._config

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}:\n{self._config}'

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


class ModelConfig(Config):
    ASSOCIATION_MODES = [None, 'GAT', 'GC', 'GCN', 'INA']
    RNN_MODES = ['GRU', 'LSTM', 'RNN']
    DISTANCE_UNITS = ['m', 'km']

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.association_mode = self._get_association_mode()
        self.association_channels = self._get_association_channels()
        self.rnn_mode = self._get_rnn_mode()
        self.rnn_hidden_units = self._get_rnn_hidden_units()
        self.input_time_steps = self._get_input_time_steps()
        self.predict_interval = self._get_predict_interval()
        self.predict_time_steps = self._get_predict_time_steps()
        self.adjacency_threshold = self._get_adjacency_threshold()

    def _get_association_mode(self) -> Optional[str]:
        mode = self.get('association_mode')
        mode = list(mode.keys())[-1] if isinstance(mode, Dict) else mode
        assert mode in ModelConfig.ASSOCIATION_MODES, \
            f'Unknown association mode "{mode}".'
        return mode

    def _get_association_channels(self) -> Optional[int]:
        mode = self.get('association_mode')
        return (int(mode.get(self.association_mode))
                if isinstance(mode, Dict) else None)

    def _get_rnn_mode(self) -> str:
        mode = self._config['RNN_mode']
        mode = list(mode.keys())[-1] if isinstance(mode, Dict) else mode
        assert mode in ModelConfig.RNN_MODES, f'Unknown RNN mode "{mode}".'
        return mode

    def _get_rnn_hidden_units(self) -> Optional[int]:
        mode = self._config['RNN_mode']
        return (int(mode.get(self.rnn_mode))
                if isinstance(mode, Dict) else None)

    def _get_input_time_steps(self) -> int:
        return int(self._config['input_time_steps'])

    def _get_predict_interval(self) -> int:
        return int(self._config['predict_interval'])

    def _get_predict_time_steps(self) -> int:
        return int(self._config['predict_time_steps'])

    def _get_adjacency_threshold(self) -> int:
        match = re.match(r"(-?\d+)([a-zA-Z]+)",
                         self._config['adjacency_threshold'])
        num, unit = match.groups()
        num = float(num)
        unit = unit.lower()
        if unit == 'm':
            return num
        elif unit == 'km':
            return num * 1000
        else:
            raise ValueError(f'Unknown unit "{unit}" for adjacency threshold.')


class DataConfig(Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.all_cities = self._get_all_cities()
        self.attributes = self._get_attributes()
        self.targets = self._get_targets()
        self.dataset_split = self._get_dataset_split()

    def _get_all_cities(self) -> Sequence[str]:
        all_cities = self._config['all_cities']
        return [all_cities] if isinstance(all_cities, str) else all_cities

    def _get_attributes(self) -> Sequence[str]:
        attributes = self._config['attributes']
        return [attributes] if isinstance(attributes, str) else attributes

    def _get_targets(self) -> Sequence[str]:
        targets = self._config['targets']
        return [targets] if isinstance(targets, str) else targets

    def _get_dataset_split(self) -> Dict[str, float]:
        return self._config['dataset_split']

    @property
    def _dataset_ratio_sum(self) -> float:
        return sum(list(self.dataset_split.values()))

    @property
    def _training_set_ratio(self) -> float:
        return (float(self.dataset_split['training']) /
                self._dataset_ratio_sum)

    @property
    def _validation_set_ratio(self) -> float:
        return (float(self.dataset_split['validation']) /
                self._dataset_ratio_sum)

    @property
    def _test_set_ratio(self) -> float:
        return (float(self.dataset_split['test']) / self._dataset_ratio_sum)

    def get_training_size(self, len_data: int) -> int:
        return int(len_data * self._training_set_ratio)

    def get_validation_size(self, len_data: int) -> int:
        return int(len_data * self._validation_set_ratio)

    def get_test_size(self, len_data: int) -> int:
        return int(len_data * self._test_set_ratio)


class BasicConfig(Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def data_time_format(self) -> str:
        return self._config['data_time_format']

    def strptime(self, time_str: str) -> datetime.datetime:
        return datetime.datetime.strptime(time_str, self.data_time_format)

    def strftime(self, time: datetime.datetime) -> str:
        return time.strftime(self.data_time_format)

    @property
    def time_attribute_name(self) -> str:
        return self._config['time_attribute_name']

    @property
    def longitude_attribute_name(self) -> str:
        return self._config['longitude_attribute_name']

    @property
    def latitude_attribute_name(self) -> str:
        return self._config['latitude_attribute_name']

    @property
    def elevation_attribute_name(self) -> str:
        return self._config['elevation_attribute_name']

    @property
    def elevation_unit(self) -> str:
        return self._config['elevation_unit']

    @property
    def dtype(self) -> str:
        return self._config['dtype']


class LearningConfig(Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def learning_rate(self) -> float:
        return float(self._config['learning_rate'])

    @property
    def early_stopping_patience(self) -> int:
        return int(self._config['early_stopping_patience'])

    @property
    def weight_decay(self) -> float:
        return float(self._config['weight_decay'])

    @property
    def batch_size(self) -> int:
        return int(self._config['batch_size'])

    @property
    def max_epochs(self) -> int:
        return int(self._config['max_epochs'])


class PredictionConfig(Config):

    def __init__(self, path) -> None:
        super().__init__(path)

    @property
    def pollutants_attributes(self) -> Sequence[str]:
        return self._config['pollutants_attributes']

    @property
    def prediction_cities(self) -> Sequence[str]:
        return self._config['prediction_cities']

    @property
    def co_factor(self) -> float:
        return self.pollutants_attributes['CO']

    @property
    def no2_factor(self) -> float:
        return self.pollutants_attributes['NO2']

    @property
    def o3_factor(self) -> float:
        return self.pollutants_attributes['O3']

    @property
    def pm10_factor(self) -> float:
        return self.pollutants_attributes['PM10']

    @property
    def pm25_factor(self) -> float:
        return self.pollutants_attributes['PM2.5']

    @property
    def so2_factor(self) -> float:
        return self.pollutants_attributes['SO2']


class ConfigHub:
    BASIC_CONFIG_BASENAME = 'basic.yaml'
    DATA_CONFIG_BASENAME = 'data.yaml'
    LEARNING_CONFIG_BASENAME = 'learning.yaml'
    MODEL_CONFIG_BASENAME = 'model.yaml'
    PREDICTION_CONFIG_BASENAME = 'prediction.yaml'

    def __init__(self, dirname: str) -> None:
        self.basic_config_path = os.path.join(dirname,
                                              self.BASIC_CONFIG_BASENAME)
        self.data_config_path = os.path.join(dirname,
                                             self.DATA_CONFIG_BASENAME)
        self.learning_config_path = os.path.join(dirname,
                                                 self.LEARNING_CONFIG_BASENAME)
        self.model_config_path = os.path.join(dirname,
                                              self.MODEL_CONFIG_BASENAME)
        self.prediction_config_path = os.path.join(
            dirname, self.PREDICTION_CONFIG_BASENAME)

        self.basic_config = BasicConfig(self.basic_config_path)
        self.data_config = DataConfig(self.data_config_path)
        self.learning_config = LearningConfig(self.learning_config_path)
        self.model_config = ModelConfig(self.model_config_path)
        self.prediction_config = None

    def __eq__(self, other: "ConfigHub") -> bool:
        return ((self.data_config == other.data_config)
                and (self.model_config == other.model_config))

    def save(self, dirname: str) -> None:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.basic_config.save(dirname=dirname,
                               filename=self.BASIC_CONFIG_BASENAME)
        self.data_config.save(dirname=dirname,
                              filename=self.DATA_CONFIG_BASENAME)
        self.learning_config.save(dirname=dirname,
                                  filename=self.LEARNING_CONFIG_BASENAME)
        self.model_config.save(dirname=dirname,
                               filename=self.MODEL_CONFIG_BASENAME)

    @classmethod
    def from_config_dir(cls, dirname: str) -> None:
        config_hub = cls(dirname)
        config_hub.prediction_config = PredictionConfig(
            config_hub.prediction_config_path)
        return config_hub

    @classmethod
    def from_log_hub(cls, dirname: str) -> None:
        return cls(os.path.join(dirname, 'config'))

    def get_model(self, dist_mat: np.ndarray, time_features_idx: Sequence[int],
                  location_mat: np.ndarray) -> Any:
        return assoc.nn.model.Model(
            assoc_mode=self.model_config.association_mode,
            assoc_channels=self.model_config.association_channels,
            rnn_mode=self.model_config.rnn_mode,
            rnn_hidden=self.model_config.rnn_hidden_units,
            in_features=len(self.data_config.attributes) + 4,
            out_features=len(self.data_config.targets),
            input_time_steps=self.model_config.input_time_steps,
            nodes=len(self.data_config.all_cities),
            dist_mat=dist_mat,
            time_features_index=time_features_idx,
            location_mat=location_mat,
            adjacency_threshold=self.model_config.adjacency_threshold)