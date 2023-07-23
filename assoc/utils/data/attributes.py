import calendar
import datetime
import math
import pytz

from chinese_calendar import is_workday

import assoc.utils.config as config
from assoc.types import *


class TimeTransformer:

    def __init__(self, config_hub: config.ConfigHub) -> None:
        self.basic_config = config_hub.basic_config
        self.time = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

    def update_time(self, time_str: str) -> None:
        time = self.basic_config.strptime(time_str)
        self.time = time.replace(tzinfo=pytz.utc)

    @property
    def chinese_time(self) -> str:
        chinese_tz = pytz.timezone('Asia/Shanghai')
        return self.time.astimezone(chinese_tz)

    @property
    def fraction_of_year_elapsed(self) -> float:
        start_of_year = datetime.datetime(self.time.year, 1, 1, tzinfo=pytz.utc)
        year_duration = datetime.timedelta(days=365)
        time_elapsed = self.time - start_of_year
        fraction_of_year_elapsed = time_elapsed.total_seconds() / year_duration.total_seconds()
        return fraction_of_year_elapsed

    @property
    def fraction_of_year_elapsed(self) -> float:
        start_of_year = datetime.datetime(self.time.year, 1, 1, tzinfo=pytz.utc)
        days = 366 if calendar.isleap(self.time.year) else 365
        year_duration = datetime.timedelta(days=days)
        time_elapsed = self.time - start_of_year
        return time_elapsed.total_seconds() / year_duration.total_seconds()

    @property
    def fraction_of_day_elapsed(self) -> float:
        start_of_day = datetime.datetime(self.time.year, self.time.month, self.time.day, tzinfo=pytz.utc)
        time_elapsed = self.time - start_of_day
        return time_elapsed.total_seconds() / 86400

    @property
    def sin_year(self) -> float:
        return math.sin(2 * math.pi * self.fraction_of_year_elapsed)

    @property
    def cos_year(self) -> float:
        return math.cos(2 * math.pi * self.fraction_of_year_elapsed)

    @property
    def sin_day(self) -> float:
        return math.sin(2 * math.pi * self.fraction_of_day_elapsed)

    @property
    def cos_day(self) -> float:
        return math.cos(2 * math.pi * self.fraction_of_day_elapsed)

    @property
    def workday(self) -> float:
        return 1.0 if is_workday(self.chinese_time) else 0.0