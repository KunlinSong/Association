import datetime
import math

from chinese_calendar import is_workday

import assoc.utils.config as config
from assoc.types import *


class TimeTransformer:

    def __init__(self, basic_config: config.BasicConfig) -> None:
        self.basic_config = basic_config
        self.time = datetime.datetime(1970, 1, 1, 0, 0, 0)

    def update_time(self, time_str: str) -> None:
        self.time = self.basic_config.strptime(time_str)

    @property
    def chinese_time(self) -> str:
        return self.time + datetime.timedelta(hours=8)

    @property
    def year_position(self) -> float:
        delta = self.time - datetime.datetime(self.time.year, 1, 1, 0, 0, 0)
        year_delta = datetime.datetime(self.time.year + 1, 1, 1, 0, 0,
                                       0) - datetime.datetime(
                                           self.time.year, 1, 1, 0, 0, 0)
        return delta.total_seconds() / year_delta.total_seconds()

    @property
    def day_position(self) -> float:
        delta = self.time - datetime.datetime(self.time.year, self.time.month,
                                              self.time.day, 0, 0, 0)
        return delta.total_seconds() / 86400

    @property
    def sin_year(self) -> float:
        return math.sin(2 * math.pi * self.year_position)

    @property
    def cos_year(self) -> float:
        return math.cos(2 * math.pi * self.year_position)

    @property
    def sin_day(self) -> float:
        return math.sin(2 * math.pi * self.day_position)

    @property
    def cos_day(self) -> float:
        return math.cos(2 * math.pi * self.day_position)

    @property
    def workday(self) -> float:
        return 1 if is_workday(self.chinese_time) else 0