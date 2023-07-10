import datetime
import os
import re
from typing import overload

import yaml

import assoc.nn.association as association
import assoc.nn.rnn_cell as rnn_cell
from assoc.types import *


class Config:

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path) as f:
            self._config = yaml.safe_load(f)

    @overload
    def save(self, dirname: str, filename: str) -> None:
        ...

    @overload
    def save(self, path: str) -> None:
        ...

    def save(self, **kwargs) -> None:
        assert (('dirname' in kwargs and 'filename' in kwargs)
                or ('path' in kwargs)
                ), f"{self.__class__.__name__}: Expected either 'dirname' and "
        "'filename' or 'path' to be specified, but received neither."

        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = os.path.join(kwargs['dirname'], kwargs['filename'])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            yaml.dump(self._config, f, indent=4, sort_keys=False)

    def __eq__(self, other: "Config") -> bool:
        return self._config == other._config

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}:\n{self._config}'

    def __getitem__(self, key: str) -> str:
        return self._config[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._config[key] = value


class ModelConfig(Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path, filename='model.yaml')
        else:
            return super().save(path=path)
    
    @property
    def association_mode(self) -> str:
        if self._config["association_mode"] is None:
            return None
        else:
            return list(self._config["association_mode"].keys())[-1]

    @property
    def association_channels(self) -> Optional[int]:
        if self.association_mode is None:
            return None
        else:
            return int(self._config["association_mode"][self.association_mode])

    @property
    def association_layer(self) -> association.AssociationModule:
        if self.association_mode is None:
            return None
        else:
            return getattr(association, self.association_mode.upper())

    @property
    def rnn_mode(self) -> str:
        return list(self._config['RNN_mode'].keys())[-1]
    
    @property
    def rnn_hidden_units(self) -> int:
        return int(self._config['RNN_mode'][self.rnn_mode])

    @property
    def rnn_cell(self) -> rnn_cell.RNNCellModule:
        return getattr(rnn_cell,
                       f'Graph{self.rnn_mode.upper()}Cell')

    @property
    def input_time_steps(self) -> int:
        return int(self._config['input_time_steps'])

    @property
    def predict_interval(self) -> int:
        return int(self._config['predict_interval'])

    @property
    def predict_time_steps(self) -> int:
        return self._config['predict_time_steps']

    @property
    def adjacency_threshold(self) -> int:
        match = re.match(r"(-?\d+)([a-zA-Z]+)", self._config['adjacency_threshold'])
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

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path, filename='data.yaml')
        else:
            return super().save(path=path)

    @property
    def all_cities(self) -> Sequence[str]:
        return self._config['all_cities']

    @property
    def attributes(self) -> Sequence[str]:
        return self._config['attributes']

    @property
    def targets(self) -> Sequence[str]:
        return self._config['targets']
    
    @property
    def dataset_split(self) -> Dict[str, float]:
        return self._config['dataset_split']

    @property
    def _dataset_ratio_sum(self) -> float:
        return sum(self.dataset_split.values())

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
        return (float(self.dataset_split['test']) /
                self._dataset_ratio_sum)

    def get_training_size(self, len_data: int) -> int:
        return int(len_data * self._training_set_ratio)

    def get_validation_size(self, len_data: int) -> int:
        return int(len_data * self._validation_set_ratio)

    def get_test_size(self, len_data: int) -> int:
        return int(len_data * self._test_set_ratio)


class BasicConfig(Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path, filename='basic.yaml')
        else:
            return super().save(path=path)
    
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

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path, filename='learning.yaml')
        else:
            return super().save(path=path)

    @property
    def initial_learning_rate(self) -> float:
        return float(self._config['initial_learning_rate'])

    @property
    def learning_rate_decay(self) -> float:
        return float(self._config['learning_rate_decay'])

    @property
    def plateau_patience(self) -> int:
        return int(self._config['plateau_patience'])

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

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path, filename='prediction.yaml')
        else:
            return super().save(path=path)

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

    def __init__(self, config_directory: str) -> None:
        self.basic_config = BasicConfig(
            os.path.join(config_directory, 'basic.yaml'))
        self.data_config = DataConfig(
            os.path.join(config_directory, 'data.yaml'))
        self.learning_config = LearningConfig(
            os.path.join(config_directory, 'learning.yaml'))
        self.model_config = ModelConfig(
            os.path.join(config_directory, 'model.yaml'))
        self.prediction_config = PredictionConfig(
            os.path.join(config_directory, 'prediction.yaml'))

    def __eq__(self, other: "ConfigHub") -> bool:
        return ((self.data_config == other.data_config)
                and (self.model_config == other.model_config))

    def save(self, directory: str) -> None:
        if not os.path.exists:
            os.makedirs(directory)
        self.basic_config.save(directory)
        self.data_config.save(directory)
        self.learning_config.save(directory)
        self.model_config.save(directory)
        self.prediction_config.save(directory)