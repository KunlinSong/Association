import datetime
import os
import re
from typing import overload

import numpy as np
import yaml

import assoc.nn as nn
import assoc.nn.association as association
import assoc.nn.rnn_cell as rnn_cell
from assoc.types import *


class Config:
    """
    A class for loading and saving configuration files in YAML format.

    Attributes:
        _config (Dict[str, Any]): The configuration dictionary.

    Methods:
        load(path: str) -> None: Loads the configuration file from the given path.
        save(dirname: str, filename: str) -> None: Saves the configuration file to the given directory and filename.
        save(path: str) -> None: Saves the configuration file to the given path.
        __eq__(other: Config) -> bool: Compares two Config objects for equality.
        __repr__() -> str: Returns a string representation of the Config object.
        __getitem__(key: str) -> Any: Returns the value of the given key in the configuration dictionary.
        __setitem__(key: str, value: Any) -> None: Sets the value of the given key in the configuration dictionary.
    """

    SAVING_DIR_OPTIONS = ['dirname', 'filename']
    SAVING_PATH_OPTIONS = ['path']

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}

    @classmethod
    def load(cls, path: str) -> 'Config':
        """
        Loads the configuration file from the given path.

        Args:
            path (str): The path to the configuration file.

        Returns:
            Config: The Config object.
        """
        with open(path, 'r') as file:
            config = cls()
            config._config = yaml.safe_load(file)
            return config

    @overload
    def save(self, dirname: str, filename: str) -> None:
        ...

    @overload
    def save(self, path: str) -> None:
        ...

    def save(self, **kwargs: Union[str, Any]) -> None:
        """
        Saves the configuration file to the given directory and filename, or to the given path.

        Args:
            dirname (str): The directory to save the configuration file in.
            filename (str): The filename to save the configuration file as.
            path (str): The path to save the configuration file to.

        Raises:
            FileExistsError: If the file already exists.
        """
        if all(option in kwargs for option in self.SAVING_DIR_OPTIONS):
            path = os.path.join(kwargs['dirname'], kwargs['filename'])
        elif all(option in kwargs for option in self.SAVING_PATH_OPTIONS):
            path = kwargs['path']
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.exists(path):
            raise FileExistsError(f'File {path} already exists.')
        else:
            with open(path, 'w') as file:
                yaml.dump(self._config, file, indent=4, sort_keys=False)

    def __eq__(self, other: 'Config') -> bool:
        """
        Compares two Config objects for equality.

        Args:
            other (Config): The other Config object.

        Returns:
            bool: True if the two Config objects are equal, False otherwise.
        """
        return self._config == other._config

    def __repr__(self) -> str:
        """
        Returns a string representation of the Config object.

        Returns:
            str: The string representation of the Config object.
        """
        return f'{self.__class__.__name__}:\n{self._config}'

    def __getitem__(self, key: str) -> Any:
        """
        Returns the value of the given key in the configuration dictionary.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            Any: The value of the given key in the configuration dictionary.
        """
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets the value of the given key in the configuration dictionary.

        Args:
            key (str): The key to set the value for.
            value (Any): The value to set for the key.
        """
        self._config[key] = value


class ModelConfig(Config):

    MODEL_CONFIG_FILENAME = 'model.yaml'
    ASSOCIATION_MODES = [None, 'GAT', 'GC', 'GCN', 'INA']
    RNN_MODES = ['GRU', 'LSTM', 'RNN']
    DISTANCE_UNITS = ['m', 'km']

    def __init__(self) -> None:
        super().__init__()

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path,
                                filename=ModelConfig.MODEL_CONFIG_FILENAME)
        else:
            return super().save(path=path)

    @property
    def association_mode(self) -> str:
        mode = self._config.get('association_mode', None)
        if mode is None:
            return None
        else:
            assert mode in ModelConfig.ASSOCIATION_MODES, \
                f'Unknown association mode "{mode}".'
            return mode if isinstance(mode, str) else list(mode.keys())[-1]

    @property
    def association_channels(self) -> Optional[int]:
        if self.association_mode is None:
            return None
        else:
            if isinstance(self._config.get('association_mode'), str):
                return None
            else:
                return int(self._config.get('association_mode').get(self.association_mode))

    @property
    def rnn_mode(self) -> str:
        mode = self._config.get('RNN_mode', None)
        assert mode in ModelConfig.RNN_MODES, f'Unknown RNN mode "{mode}".'
        return mode if isinstance(mode, str) else list(mode.keys())[-1]

    @property
    def rnn_hidden_units(self) -> int:
        mode = self._config.get('RNN_mode')
        return None if isinstance(mode, str) else int(mode[self.rnn_mode])

    @property
    def input_time_steps(self) -> int:
        return int(self._config.get('input_time_steps'))

    @property
    def predict_interval(self) -> int:
        return int(self._config.get('predict_interval'))

    @property
    def predict_time_steps(self) -> int:
        return self._config.get('predict_time_steps')

    @property
    def adjacency_threshold(self) -> int:
        match = re.match(r"(-?\d+)([a-zA-Z]+)",
                         self._config.get('adjacency_threshold'))
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

    def __init__(self) -> None:
        super().__init__()

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
        return (float(self.dataset_split['test']) / self._dataset_ratio_sum)

    def get_training_size(self, len_data: int) -> int:
        return int(len_data * self._training_set_ratio)

    def get_validation_size(self, len_data: int) -> int:
        return int(len_data * self._validation_set_ratio)

    def get_test_size(self, len_data: int) -> int:
        return int(len_data * self._test_set_ratio)


class BasicConfig(Config):

    def __init__(self) -> None:
        super().__init__()

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

    def __init__(self) -> None:
        super().__init__()

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            return super().save(dirname=path, filename='learning.yaml')
        else:
            return super().save(path=path)

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

    def __init__(self) -> None:
        super().__init__()

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

    def __init__(self) -> None:
        self.basic_config = BasicConfig()
        self.data_config = DataConfig()
        self.learning_config = LearningConfig()
        self.model_config = ModelConfig()
        self.prediction_config = PredictionConfig()

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

    @overload
    def load(self, config_hub: 'ConfigHub') -> None:...
    
    @overload
    def load(self, dir: str) -> None:...

    def load(self, *args) -> None:
        if isinstance(args[0], str):
            self._load_from_dir(args[0])
        elif isinstance(args[0], ConfigHub):
            self._load_from_config_hub(args[0])
    
    def _load_from_dir(self, dir: str) -> None:
        self.basic_config = BasicConfig.load(
            os.path.join(dir, 'basic.yaml'))
        self.data_config = DataConfig.load(
            os.path.join(dir, 'data.yaml'))
        self.learning_config = LearningConfig.load(
            os.path.join(dir, 'learning.yaml'))
        self.model_config = ModelConfig.load(
            os.path.join(dir, 'model.yaml'))
        self.prediction_config = PredictionConfig.load(
            os.path.join(dir, 'prediction.yaml'))

    def _load_from_config_hub(self, config_hub: 'ConfigHub') -> None:
        self = config_hub
    
    def get_model(self, dist_mat: np.ndarray, time_features_idx: Sequence[int], location_mat: np.ndarray) -> Any:
        return nn.model.Model(self.model_config.association_mode,
                              self.model_config.association_channels,
                              self.model_config.rnn_mode,
                              self.model_config.rnn_hidden_units,
                              len(self.data_config.attributes) + 4,
                              len(self.data_config.targets),
                              self.model_config.input_time_steps,
                              len(self.data_config.all_cities),
                              dist_mat, time_features_idx, location_mat,
                              self.model_config.adjacency_threshold)