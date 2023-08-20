import datetime
import math
import os

import numpy as np
import pandas as pd
import pytz
from geopy import Point, distance

from association.utils.config.confighub import ConfigHub
import association.utils.data.attribute as attribute
from association.types import *


class CityData:

    def __init__(self, path: str, config_hub: ConfigHub) -> None:
        self._data = pd.read_csv(path)
        self._config_hub = config_hub
        self._attributes = self._config_hub.input_attributes.copy()
        if (config_hub.time_attribute_name
                in self._config_hub.input_attributes):
            _time_transformer = attribute.TimeTransformer(config_hub)
            self._data = _time_transformer(self._data)
            self._attributes.extend(_time_transformer.TIME_ATTRIBUTES)
            self._attributes.remove(config_hub.time_attribute_name)
        self.num_attributes = len(self._attributes)
        self.pollutants_idx = None

    # TODO: 两个获取数据的函数不是按照时间步返回，需要在dataset或data进一步处理。
    # TODO: Dataset最好需要有train, valid, test, predict四种模式。
    def get_input_data(self, time_lst: list[str]) -> np.ndarray:
        return self._data[self._data[self._config_hub.time_attribute_name].
                          isin(time_lst)][self._attributes].values

    def get_target_data(self, time_lst: list[str]) -> np.ndarray:
        return self._data[self._data[self._config_hub.time_attribute_name].
                          isin(time_lst)][self._config_hub.targets].values

    def get_predict_input(self, time_lst: list[str]) -> np.ndarray:
        data = self._data[self._data[self._config_hub.time_attribute_name].
                          isin(time_lst)][self._attributes].copy()
        for pollutant, factor in self._config_hub.pollutants_modify_dict.items(
        ):
            data[pollutant] *= factor
        return data.values


class Location:

    def __init__(self, location: List[float]) -> None:
        self.latitude = location[0]
        self.longitude = location[1]
        self.elevation = location[2]
        self.point = Point(self.latitude, self.longitude)

    def distance_to(self, other: 'Location') -> float:
        flat_dist = distance.distance(self.point, other.point).m
        elevation_diff = abs(self.elevation - other.elevation)
        return math.sqrt(flat_dist**2 + elevation_diff**2)


class LocationCollection:

    def __init__(self, location_csv: str, config_hub: ConfigHub) -> None:
        df = pd.read_csv(location_csv, index_col=0)
        self.config_hub = config_hub
        self.locations = df.loc[config_hub.input_cities, [
            config_hub.latitude_attribute_name, config_hub.
            longitude_attribute_name, config_hub.elevation_attribute_name
        ]].values
        self.distance_matrix = self._get_distance_matrix()

    def _get_distance_matrix(self) -> np.ndarray:
        city_num = len(self.config_hub.input_cities)
        distance_matrix = np.zeros((city_num, city_num))
        locations = self.locations.tolist()
        for i in range(city_num):
            for j in range(city_num):
                if i != j:
                    location_i = Location(locations[i])
                    location_j = Location(locations[j])
                    distance_matrix[i][j] = location_i.distance_to(location_j)
        return distance_matrix


class Data:

    def __init__(self, dirname: str, config_hub: ConfigHub) -> None:
        self._config_hub = config_hub
        self.cities_path = {
            f'{city}': os.path.join(dirname, city, f'{city}.csv')
            for city in config_hub.input_cities
        }
        self._cities_data = {
            city: CityData(path, config_hub)
            for city, path in self.cities_path.items()
        }
        self._location_path = os.path.join(dirname, 'location.csv')
        self._location_collection = LocationCollection(self._location_path,
                                                       config_hub)
        self.num_attributes = self._cities_data[list(self._cities_data.keys())[0]].num_attributes

    @property
    def distance_matrix(self) -> np.ndarray:
        return self._location_collection.distance_matrix

    # TODO: shape: time, city, attribute
    def get_input_data(self, time_lst: list[str]) -> np.ndarray:
        input_data_lst = [
            self._cities_data[city].get_input_data(time_lst)
            for city in self._config_hub.input_cities
        ]
        return np.stack(input_data_lst, axis=1)

    def get_target_data(self, time_lst: list[str]) -> np.ndarray:
        target_data_lst = [
            self._cities_data[city].get_target_data(time_lst)
            for city in self._config_hub.input_cities
        ]
        return np.stack(target_data_lst, axis=1)

    # Only works when config hub is for predict.
    def get_predict_input(self, time_lst: list[str]) -> np.ndarray:
        predict_input_lst = [
            (self._cities_data[city].get_predict_input(time_lst) * 
             self._config_hub.cities_modify_dict[city])
            for city in self._config_hub.input_cities
        ]
        return np.stack(predict_input_lst, axis=1)