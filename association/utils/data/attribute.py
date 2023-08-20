import calendar
import datetime
import math
import pytz

import pandas as pd
from chinese_calendar import is_workday

from association.types import *
from association.utils.config.confighub import ConfigHub


class TimeTransformer:
    TIME_ATTRIBUTES = ['sin_year', 'cos_year', 'sin_day', 'cos_day', 'workday']

    def __init__(self, config_hub: ConfigHub) -> None:
        self._config_hub = config_hub

    def update_time(self, time_str: str) -> None:
        time = self._config_hub.strptime(time_str=time_str)
        self._time = time.replace(tzinfo=pytz.utc)

    @property
    def _chinese_time(self) -> str:
        chinese_tz = pytz.timezone('Asia/Shanghai')
        return self._time.astimezone(chinese_tz)

    @property
    def _fraction_of_year_elapsed(self) -> float:
        start_of_year = datetime.datetime(self._time.year,
                                          1,
                                          1,
                                          tzinfo=pytz.utc)
        year_duration = datetime.timedelta(days=365)
        time_elapsed = self._time - start_of_year
        fraction_of_year_elapsed = time_elapsed.total_seconds(
        ) / year_duration.total_seconds()
        return fraction_of_year_elapsed

    @property
    def _fraction_of_year_elapsed(self) -> float:
        start_of_year = datetime.datetime(self._time.year,
                                          1,
                                          1,
                                          tzinfo=pytz.utc)
        days = 366 if calendar.isleap(self._time.year) else 365
        year_duration = datetime.timedelta(days=days)
        time_elapsed = self._time - start_of_year
        return time_elapsed.total_seconds() / year_duration.total_seconds()

    @property
    def _fraction_of_day_elapsed(self) -> float:
        start_of_day = datetime.datetime(self._time.year,
                                         self._time.month,
                                         self._time.day,
                                         tzinfo=pytz.utc)
        time_elapsed = self._time - start_of_day
        return time_elapsed.total_seconds() / 86400

    @property
    def sin_year(self) -> float:
        return math.sin(2 * math.pi * self._fraction_of_year_elapsed)

    @property
    def cos_year(self) -> float:
        return math.cos(2 * math.pi * self._fraction_of_year_elapsed)

    @property
    def sin_day(self) -> float:
        return math.sin(2 * math.pi * self._fraction_of_day_elapsed)

    @property
    def cos_day(self) -> float:
        return math.cos(2 * math.pi * self._fraction_of_day_elapsed)

    @property
    def workday(self) -> float:
        return 1.0 if is_workday(self._chinese_time) else 0.0
    
    def _get_attributes(self, time: str) -> pd.Series:
        self.update_time(time)
        return pd.Series([
            self.sin_year, self.cos_year, self.sin_day, self.cos_day,
            self.workday
        ])

    def __call__(self, df: pd.DataFrame) -> Any:
        time_attribute_name = self._config_hub.time_attribute_name
        df[self.TIME_ATTRIBUTES] = df[time_attribute_name].apply(
                self._get_attributes)
        return df