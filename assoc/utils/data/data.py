import datetime
import os
from typing import overload

import numpy as np
import pandas as pd
import pytz
from geopy import Point, distance

import assoc.utils.config as config
import assoc.utils.data.attributes as attributes
from assoc.types import *


__all__ = ['CityData', 'Data', 'Location', 'LocationCollection', 'TimeDict']


class TimeDict:

    def __init__(self, data_dir: str, config_hub: config.ConfigHub) -> None:
        self.data_dir = data_dir
        self.config_hub = config_hub
        self.time_set = self._get_common_data_time()
        self.start_time = min(self.time_set)
        self.end_time = max(self.time_set)
        self.time_step = min(self.time_set -
                             set([self.start_time])) - self.start_time
        self.input_time_map, self.predict_time_map = self._get_time_delta_map()
        self.time_dict = self._get_time_dict()

    def _get_common_data_time(self) -> set[datetime.datetime]:
        time_set = None
        for city in self.config_hub.data_config.all_cities:
            csv_path = os.path.join(self.data_dir, city, f'{city}.csv')
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=[self.config_hub.data_config.attributes])
            time_str_lst = df[
                self.config_hub.basic_config.time_attribute_name].tolist()
            time_lst = [
                self.config_hub.basic_config.strptime(time_str)
                for time_str in time_str_lst
            ]
            time_lst = [time.replace(tzinfo=pytz.utc) for time in time_lst]
            data_time_set = set(time_lst)
            if time_set is None:
                time_set = data_time_set
            else:
                time_set &= data_time_set
        return time_set

    def _get_time_delta_map(
            self) -> tuple[list[datetime.timedelta], list[datetime.timedelta]]:
        input_time_map = [
            self.time_step * i
            for i in range(self.config_hub.model_config.input_time_steps)
        ]
        predict_time_map = [
            self.time_step *
            (i + self.config_hub.model_config.predict_interval)
            for i in range(self.config_hub.model_config.input_time_steps +
                           self.config_hub.model_config.predict_time_steps)
        ]
        return input_time_map, predict_time_map

    def _get_time_dict(self) -> Dict[str, tuple[list[str], list[str]]]:
        time_dict = {}
        position = self.start_time
        while position <= self.end_time:
            input_time = [position + delta for delta in self.input_time_map]
            predict_time = [
                position + delta for delta in self.predict_time_map
            ]
            if set(input_time).issubset(
                    self.time_set) and set(predict_time).issubset(
                        self.time_set):
                input_time = [
                    self.config_hub.basic_config.strftime(time)
                    for time in input_time
                ]
                predict_time = [
                    self.config_hub.basic_config.strftime(time)
                    for time in predict_time
                ]
                time_dict[self.config_hub.basic_config.strftime(position)] = (
                    input_time, predict_time)
            position += self.time_step
        return time_dict

    def keys(self) -> list[str]:
        return self.time_dict.keys()

    def __getitem__(self, time: str) -> tuple[list[str], list[str]]:
        return self.time_dict[time]


class Location:

    @overload
    def __init__(self, latitude: float, longitude: float,
                 elevation: float) -> None:
        ...

    @overload
    def __init__(self, location: Sequence[float]) -> None:
        ...

    def __init__(self, *args) -> None:
        self.latitude = args[0]
        self.longitude = args[1]
        self.elevation = args[2]
        self.point = Point(self.latitude, self.longitude, self.elevation)

    def distance_to(self, other: 'Location') -> float:
        return distance.distance(self.point, other.point).m


class LocationCollection:

    def __init__(self, location_csv: str,
                 config_hub: config.ConfigHub) -> None:
        df = pd.read_csv(location_csv)
        self.locations = df[config_hub.data_config.all_cities, [
            config_hub.basic_config.latitude_attribute_name, config_hub.
            basic_config.longitude_attribute_name, config_hub.basic_config.
            elevation_attribute_name
        ]].values
        self.elevation_unit = config_hub.basic_config.elevation_unit
        self.distance_matrix = self._get_distance_matrix()

    def _get_distance_matrix(self) -> np.ndarray:
        locations_1 = self.locations.repeat(len(self.locations), axis=0)
        locations_2 = np.tile(self.locations, (len(self.locations), 1))
        city_num = self.locations.shape[0]
        distance_lst = []
        for i, j in zip(locations_1, locations_2):
            location_i = Location(i)
            location_j = Location(j)
            distance_lst.append(location_i.distance_to(location_j))
        return np.array(distance_lst).reshape(city_num, city_num)


class Data:

    def __init__(self, data: pd.DataFrame,
                 time_transformer: attributes.TimeTransformer) -> None:
        self.data = data.copy()
        self.time_attributes = [
            'sin_year', 'cos_year', 'sin_day', 'cos_day', 'workday'
        ]
        self.time_transformer = time_transformer
        self.data[self.time_attributes] = self.data[
            self.time_transformer.basic_config.time_attribute_name].apply(
                self._get_value)
        self.data = self.data.drop(columns=['time'])

    def _get_value(self, time) -> float:
        self.time_transformer.update_time(time)
        values = [
            getattr(self.time_transformer, attribute)()
            for attribute in self.time_attributes
        ]
        return values


class CityData:

    def __init__(self, csv_path: str, time_dict: TimeDict) -> None:
        self.csv_data = pd.read_csv(csv_path)
        self.time_dict = time_dict
        self.time_transformer = attributes.TimeTransformer(
            time_dict.config_hub)
        self.city_data = Data(self.csv_data, self.time_transformer)

    def __getitem__(self, time: str) -> tuple[np.ndarray, np.ndarray]:
        input_time, predict_time = self.time_dict[time]
        input_data = self.city_data.data[self.csv_data['time'].isin(
            input_time)].values
        predict_data = self.city_data.data[self.csv_data['time'].isin(
            predict_time)].values
        return input_data, predict_data