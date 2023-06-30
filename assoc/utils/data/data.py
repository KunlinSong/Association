import datetime
import math
import os

from geopy import distance
import numpy as np
import pandas as pd

import assoc.utils.config as config
from assoc.types import *


class TimeDict:

    def __init__(self, data_dir: str, config_hub: config.ConfigHub) -> None:
        self.data_dir = data_dir
        self.config_hub = config_hub
        self.time_set = self._get_data_time()
        self.start = min(self.time_set)
        self.end = max(self.time_set)
        self.step_delta = min(self.time_set - set([self.start])) - self.start
        self.input_map, self.predict_map = self._get_time_delta_map()
        self.time_dict = self._get_time_dict()

    def _get_data_time(self) -> set:
        time_set = None
        for city in self.config_hub.data_config.all_cities:
            csv_path = os.path.join(self.data_dir, city, f'{city}.csv')
            df = pd.read_csv(csv_path)
            time_str_lst = df[self.config_hub.basic_config.time_attributes_name].to_list()
            time_str_lst = [
                self.config_hub.basic_config.strptime(time_str)
                for time_str in time_str_lst
            ]
            nan_time_str_lst = df[df[self.config_hub.data_config.attributes].isnull().any(
                axis=1) == True][
                    self.config_hub.basic_config.time_attributes_name].to_list()
            nan_time_str_lst = [
                self.config_hub.basic_config.strptime(time_str)
                for time_str in nan_time_str_lst
            ]
            data_time_set = set(time_str_lst) - set(nan_time_str_lst)
            if time_set is None:
                time_set = data_time_set
            else:
                time_set = time_set & data_time_set
        return time_set

    def _get_time_delta_map(self) -> tuple[list, list]:
        input_map = [
            self.step_delta * i
            for i in range(self.config_hub.model_config.input_time_steps)
        ]
        predict_start = self.step_delta * (self.config_hub.model_config.input_time_steps +
                                           self.config_hub.model_config.predict_interval)
        predict_map = [(predict_start + self.step_delta * i)
                       for i in range(self.config_hub.model_config.predict_time_steps)]
        return input_map, predict_map

    def _get_time_dict(self) -> dict:
        time_dict = {}
        position = self.start
        while position <= self.end:
            input_time = [position + delta for delta in self.input_map]
            predict_time = [position + delta for delta in self.predict_map]
            if set(input_time).issubset(
                    self.time_set) and set(predict_time).issubset(
                        self.time_set):
                input_time = [
                    self.config_hub.basic_config.strftime(time) for time in input_time
                ]
                predict_time = [
                    self.config_hub.basic_config.strftime(time) for time in predict_time
                ]
                time_dict[self.config_hub.basic_config.strftime(position)] = (
                    input_time, predict_time)
            position += self.step_delta
        return time_dict
    
    def keys(self) -> list:
        return list(self.time_dict.keys())
    
    def __getitem__(self, time: str) -> tuple[list, list]:
        return self.time_dict[time]


class Location:

    def __init__(self,
                 location_csv: str,
                 config_hub: config.ConfigHub,
                 elevation_unit: Literal['km', 'm'] = 'm') -> None:
        df = pd.read_csv(location_csv)
        self.location_matrix = df[config_hub.data_config.all_cities, [
            config_hub.basic_config.latitude_attribute_name, config_hub.basic_config.
            longitude_attribute_name, config_hub.basic_config.elevation_attribute_name
        ]].values
        self.elevation_unit = elevation_unit

    def _distance(self,
                  i_coordinate: float,
                  j_coordinate: float,
                  distance_unit: Literal['km', 'm'] = 'km') -> float:
        distance_flat = distance.distance(i_coordinate, j_coordinate).m
        distance_elevation = (i_coordinate[2] - j_coordinate[2]) if (
            self.elevation_unit
            == 'm') else ((i_coordinate[2] - j_coordinate[2]) * 1000)
        distance = math.sqrt(distance_flat**2 + distance_elevation**2)
        return distance if (distance_unit == 'm') else (distance / 1000)

    def distance_matrix(self,
                        distance_unit: Literal['km',
                                               'm'] = 'km') -> np.ndarray:
        co1 = self.location_matrix.repeat(len(self.location_matrix), axis=0)
        co2 = np.tile(self.location_matrix, (len(self.location_matrix), 1))
        city_num = self.location_matrix.shape[0]
        distance_array = np.zeros((city_num, city_num))
        for num, (i_co, j_co) in enumerate(zip(co1, co2)):
            distance_ij = self._distance(i_co, j_co, self.elevation_unit,
                                         distance_unit)
            distance_array[num // city_num, num % city_num] = distance_ij
        return distance_array


class CityData:

    def __init__(self, csv_path: str, time_dict: TimeDict) -> None:
        self.csv_data = pd.read_csv(csv_path)
        self.time_dict = time_dict
    
    def __getitem__(self, time: str) -> pd.DataFrame:
        input_time, predict_time = self.time_dict[time]
        input_data = self.csv_data[self.csv_data[
            self.time_dict.basic_config.time_attributes_name].isin(
                input_time)]
        predict_data = self.csv_data[self.csv_data[
            self.time_dict.basic_config.time_attributes_name].isin(
                predict_time)]
        return Data(input_data, predict_data)

# TODO: finish this class
class Data:
    def __init__(self) -> None:
        pass