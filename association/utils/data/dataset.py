import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from association.utils.config.confighub import ConfigHub
from association.types import *
from association.utils.data.data import Data
from association.utils.data.assistant import TimeDict


class Dataset(data.Dataset):

    def __init__(self, dirname: str, config_hub: ConfigHub) -> None:
        super().__init__()
        self.dirname = dirname
        self.config_hub = config_hub
        self.data = Data(dirname, config_hub)
        self.dtype = getattr(torch, self.config_hub.dtype)
        self.time_dict = TimeDict(dirname, config_hub)
        self.state = None
        self.training_size = self.config_hub.get_training_size(
            len(self.time_dict))
        self.validation_size = self.config_hub.get_validation_size(
            len(self.time_dict))
        self.test_size = self.config_hub.get_test_size(len(self.time_dict))
        self.num_attributes = self.data.num_attributes

    @property
    def distance_matrix(self) -> np.ndarray:
        return self.data.distance_matrix

    @property
    def graph(self) -> torch.Tensor:
        return torch.from_numpy((self.distance_matrix > 0) & (
            self.distance_matrix <= self.config_hub.threshold)).to(self.dtype)

    def to_state(self, state: Literal['training', 'validation',
                                      'test']) -> None:
        self.state = state

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        match self.state:
            case 'training':
                key = f'{idx}'
            case 'validation':
                key = f'{idx + self.training_size}'
            case 'test':
                key = f'{idx + self.training_size + self.validation_size}'
            case _:
                raise ValueError(f'Invalid state: {self.state}')
        input_time, target_time = self.time_dict[key]
        input_data = self.data.get_input_data(input_time)
        target_data = self.data.get_target_data(target_time)

        if self.config_hub.predict_average != 1:
            target_data = np.stack([
                np.mean(np.stack(target_data[i:i +
                                             self.config_hub.predict_average]),
                        axis=0) for i in range(self.config_hub.input_time_step)
            ])
        else:
            target_data = np.stack(target_data)

        input_data = torch.from_numpy(input_data).to(self.dtype)
        target_data = torch.from_numpy(target_data).to(self.dtype)
        return input_data, target_data

    def __len__(self):
        match self.state:
            case 'training':
                return self.training_size
            case 'validation':
                return self.validation_size
            case 'test':
                return self.test_size
            case _:
                raise ValueError(f'Invalid state: {self.state}')


class PredictDataset(data.Dataset):
    def __init__(self, dirname: str, config_hub: ConfigHub) -> None:
        super().__init__()
        self.dirname = dirname
        self.config_hub = config_hub
        self.data = Data(dirname, config_hub)
        self.dtype = getattr(torch, self.config_hub.dtype)
        self.time_dict = TimeDict(dirname, config_hub, predict=True)
    
    def __len__(self):
        return len(self.time_dict)
    
    @property
    def distance_matrix(self) -> np.ndarray:
        return self.data.distance_matrix

    @property
    def graph(self) -> torch.Tensor:
        return torch.from_numpy((self.distance_matrix > 0) & (
            self.distance_matrix <= self.config_hub.threshold)).to(self.dtype)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        input_time, target_time = self.time_dict[f'{idx}']
        input_data = self.data.get_predict_input(input_time)
        target_data = self.data.get_target_data(target_time)

        if self.config_hub.predict_average != 1:
            target_data = np.stack([
                np.mean(np.stack(target_data[i:i +
                                             self.config_hub.predict_average]),
                        axis=0) for i in range(self.config_hub.input_time_step)
            ])
        else:
            target_data = np.stack(target_data)

        input_data = torch.from_numpy(input_data).to(self.dtype)
        target_data = torch.from_numpy(target_data).to(self.dtype)
        return input_data, target_data
    

class AssociationMatrixDataset:
    def __init__(self, dirname: str, config_hub: ConfigHub) -> None:
        self.dirname = dirname
        self.config_hub = config_hub
        self.data = Data(dirname, config_hub)
        self.dtype = getattr(torch, self.config_hub.dtype)
    
    # TODO: The shape of input_data is (city, attribute).
    def get_input_data(self, time_lst: list[str]) -> torch.Tensor:
        time_set = set(time_lst)
        for city in self.config_hub.input_cities:
            csv_path = os.path.join(self.dirname, city, f'{city}.csv')
            df = pd.read_csv(csv_path)
            time_str_lst = df.loc[
                df[self.config_hub.input_attributes].notnull().all(axis=1),
                self.config_hub.time_attribute_name].tolist()
            time_set &= set(time_str_lst)
        time_lst = list(time_set)
        input_data = self.data.get_input_data(time_lst)
        if len(time_lst) > 1:
            input_data = np.mean(input_data, axis=0)
        input_data = torch.from_numpy(input_data).to(self.dtype)
        return input_data