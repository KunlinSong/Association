import datetime
import os

import pandas as pd
import pytz

from association.utils.config.confighub import ConfigHub
from association.types import *


class TimeDict:

    def __init__(self, data_dir: str, config_hub: ConfigHub, predict: bool=False) -> None:
        self.data_dir = data_dir
        self.config_hub = config_hub
        self.time_interval = datetime.timedelta(hours=config_hub.time_interval)
        self.input_map, self.target_map = self._get_time_delta_map()
        self.time_set = self._get_common_data_time()
        if predict:
            self.start_time, self.end_time = self.config_hub.range_datetimes
        else:
            self.start_time = min(self.time_set)
            self.end_time = max(self.time_set)
        self.time_dict = self._get_time_dict()

    def _get_common_data_time(self) -> set[datetime.datetime]:
        time_set = None
        for city in self.config_hub.input_cities:
            csv_path = os.path.join(self.data_dir, city, f'{city}.csv')
            df = pd.read_csv(csv_path)

            time_str_lst = df.loc[
                df[self.config_hub.input_attributes].notnull().all(axis=1),
                self.config_hub.time_attribute_name].tolist()
            time_set = (set(time_str_lst) if time_set is None else
                        (time_set & set(time_str_lst)))
        time_lst = list(time_set)
        time_lst = [
            self.config_hub.strptime(time_str).replace(tzinfo=pytz.utc)
            for time_str in time_lst
        ]
        return set(time_lst)

    def _get_time_delta_map(
            self) -> tuple[list[datetime.timedelta], list[datetime.timedelta]]:
        input_map = [
            self.time_interval * i
            for i in range(self.config_hub.input_time_step)
        ]
        target_map = [
            (self.time_interval * (i + self.config_hub.predict_interval))
            for i in range(self.config_hub.input_time_step +
                           self.config_hub.predict_average)
        ]
        return input_map, target_map

    def _get_time_dict(self) -> Dict[str, tuple[list[str], list[str]]]:
        time_dict = {}
        position = self.start_time
        i = 0
        while position <= self.end_time:
            input_time = [position + delta for delta in self.input_map]
            predict_time = [position + delta for delta in self.target_map]
            if set(input_time).issubset(
                    self.time_set) and set(predict_time).issubset(
                        self.time_set):
                input_time = [
                    self.config_hub.strftime(time) for time in input_time
                ]
                predict_time = [
                    self.config_hub.strftime(time) for time in predict_time
                ]
                time_dict[f'{i}'] = (input_time, predict_time)
                i += 1
            position += self.time_interval
        return time_dict

    def keys(self) -> list[str]:
        return self.time_dict.keys()

    def __getitem__(self, idx: str) -> tuple[list[str], list[str]]:
        return self.time_dict[idx]

    def __len__(self) -> int:
        return len(self.time_dict)