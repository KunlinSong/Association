import os

import numpy as np
import torch
import torch.utils.data as data

import assoc.utils.config as config
from assoc.types import *
from assoc.utils.data.data import CityData, LocationCollection, TimeDict


class Dataset(data.Dataset):

    def __init__(self, dir: str, config_hub: config.ConfigHub) -> None:
        super().__init__()
        self.dir = dir
        self.config_hub = config_hub
        self.dtype = getattr(torch, self.config_hub.basic_config.dtype)
        self.time_dict = TimeDict(dir, config_hub)
        self.prediction_cities_idx_lst = self._get_prediction_cities_idx_lst()
        self.city_data_lst = self._get_city_data_lst()
        self.training_idx, self.validation_idx, self.test_idx = self._get_keys(
        )
        self.state = None
        self.dataset = None
        self.time_attr_idx = self.city_data_lst[0].time_attr_idx

    def to_state(self, state: Literal['training', 'validation',
                                      'test']) -> None:
        self.state = state

    def _get_prediction_cities_idx_lst(self) -> list[int]:
        all_cities = self.config_hub.data_config.all_cities
        prediction_cities = self.config_hub.prediction_config.prediction_cities
        prediction_cities_idx_lst = [
            all_cities.index(city) for city in prediction_cities
        ]
        return prediction_cities_idx_lst

    def _get_city_data_lst(self) -> list[CityData]:
        city_data_lst = []
        for city in self.config_hub.data_config.all_cities:
            csv_path = os.path.join(self.dir, city, f'{city}.csv')
            city_data = CityData(csv_path=csv_path, time_dict=self.time_dict)
            city_data_lst.append(city_data)
        return city_data_lst

    def _get_keys(self) -> tuple[list[str], list[str], list[str]]:
        all_keys = self.time_dict.keys()
        keys_num = len(all_keys)
        training_size = self.config_hub.data_config.get_training_size(keys_num)
        validation_size = self.config_hub.data_config.get_validation_size(
            keys_num)
        training_validation_idx = training_size
        validation_test_idx = training_size + validation_size
        return (all_keys[:training_validation_idx],
                all_keys[training_validation_idx:validation_test_idx],
                all_keys[validation_test_idx:])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state == 'training':
            key = self.training_idx[idx]
        elif self.state == 'validation':
            key = self.validation_idx[idx]
        elif self.state == 'test':
            key = self.test_idx[idx]
        else:
            raise ValueError(f'Invalid state: {self.state}')
        city_data_lst = [city_data[key] for city_data in self.city_data_lst]
        input_lst = [city_data[0] for city_data in city_data_lst]
        target_lst = [city_data[1] for city_data in city_data_lst]
        target_data = []
        for i in range(self.config_hub.model_config.input_time_steps):
            add_data = target_lst[i:i + self.config_hub.model_config.
                                  predict_time_steps]
            target_data.append(np.mean(np.array(add_data), axis=0))
        input_data = torch.tensor(input_lst, dtype=self.dtype)
        target_data = torch.tensor(target_data, dtype=self.dtype)
        return input_data, target_data

    def __len__(self):
        if self.state == 'training':
            return len(self.training_idx)
        elif self.state == 'validation':
            return len(self.validation_idx)
        elif self.state == 'test':
            return len(self.test_idx)
        else:
            raise ValueError(f'Invalid state: {self.state}')