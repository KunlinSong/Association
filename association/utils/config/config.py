import datetime
import os

import yaml

from association.types import *

__all__ = [
    'AssociationMatrixConfig', 'BasicConfig', 'DataConfig', 'EvaluateConfig', 'LearningConfig',
    'ModelConfig', 'PredictConfig'
]


class _Config:

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Config file not found in {path}')
        self._path = path
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)

    @property
    def path(self) -> str:
        return self._path

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.safe_dump(self._config, f, sort_keys=False)

    def __eq__(self, other: '_Config') -> bool:
        return self._config == other._config


class AssociationMatrixConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def start(self) -> datetime.datetime:
        info = self._config.get('Start')
        if info is None:
            raise ValueError(f'Start time not found in {self.path}')
        else:
            return datetime.datetime(**info)

    @property
    def end(self) -> datetime.datetime:
        info = self._config.get('End')
        if info is None:
            raise ValueError(f'End time not found in {self.path}')
        else:
            return datetime.datetime(**info)

    @property
    def segment_start(self) -> datetime.time:
        info = self._config.get('TimeSegment').get('Start')
        if info is None:
            raise ValueError(f'Segment start time not found in {self.path}')
        else:
            return datetime.time(**info)

    @property
    def segment_end(self) -> datetime.time:
        info = self._config.get('TimeSegment').get('End')
        if info is None:
            raise ValueError(f'Segment end time not found in {self.path}')
        else:
            return datetime.time(**info)

    @property
    def segment_datetimes(self) -> list[datetime.datetime]:
        start = self.start
        end = self.end
        segment_start = self.segment_start
        segment_end = self.segment_end
        day_state = start.date()
        end_date = end.date()
        day_time_delta = datetime.timedelta(days=1)
        segment_time_delta = datetime.timedelta(hours=1)
        segment_datetimes = []
        while day_state <= end_date:
            day_start = datetime.datetime.combine(day_state, segment_start)
            day_end = datetime.datetime.combine(day_state, segment_end)
            segment_state = day_start
            while segment_state <= day_end:
                if start <= segment_state <= end:
                    segment_datetimes.append(segment_state)
                segment_state += segment_time_delta
            day_state += day_time_delta
        return segment_datetimes


class BasicConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def time_format(self) -> str:
        return self._config.get('TimeFormat', '%Y-%m-%d %H:%M:%S')

    def strptime(self, time_str: str) -> datetime.datetime:
        return datetime.datetime.strptime(time_str, self.time_format)

    def strftime(self, time: datetime.datetime) -> str:
        return time.strftime(self.time_format)

    @property
    def time_attribute_name(self) -> str:
        return self._config.get('TimeAttribute', 'time')

    @property
    def longitude_attribute_name(self) -> str:
        return self._config.get('LongitudeAttribute', 'longitude')

    @property
    def latitude_attribute_name(self) -> str:
        return self._config.get('LatitudeAttribute', 'latitude')

    @property
    def elevation_attribute_name(self) -> str:
        return self._config.get('ElevationAttribute', 'elevation')

    @property
    def dtype(self) -> str:
        return self._config.get('DType', 'float32')


class DataConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def input_cities(self) -> list[str]:
        return self._config.get('InputCities', [])

    @property
    def input_attributes(self) -> list[str]:
        return self._config.get('InputAttributes', [])

    @property
    def targets(self) -> list[str]:
        return self._config.get('Targets', [])

    @property
    def _dataset_split(self) -> dict[str, float]:
        return self._config.get('DatasetSplit', {
            'Training': 0.6,
            'Validation': 0.2,
            'Test': 0.2
        })

    @property
    def _split_sum(self) -> float:
        return sum(self._dataset_split.values())

    @property
    def training_set_ratio(self) -> float:
        return self._dataset_split['Training'] / self._split_sum

    @property
    def validation_set_ratio(self) -> float:
        return self._dataset_split['Validation'] / self._split_sum

    @property
    def test_set_ratio(self) -> float:
        return self._dataset_split['Test'] / self._split_sum

    def get_training_size(self, total_size: int) -> int:
        return int(total_size * self.training_set_ratio)

    def get_validation_size(self, total_size: int) -> int:
        return int(total_size * self.validation_set_ratio)

    def get_test_size(self, total_size: int) -> int:
        return int(total_size * self.test_set_ratio)


class LearningConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def batch_size(self) -> int:
        return self._config.get('BatchSize', 64)

    @property
    def learning_rate(self) -> float:
        return float(self._config.get('LearningRate', 0.01))

    @property
    def weight_decay(self) -> float:
        return float(self._config.get('WeightDecay', 1e-4))

    @property
    def max_epoch(self) -> int:
        return self._config.get('MaxEpoch', 100)

    @property
    def early_stopping_patience(self) -> int:
        return self._config.get('EarlyStoppingPatience', 20)


class ModelConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def association_mode(self) -> str:
        mode = self._config.get('AssociationMode')
        return list(mode.keys())[-1] if isinstance(mode, dict) else mode

    @property
    def association_param(self) -> int:
        mode = self._config.get('AssociationMode')
        return mode[self.association_mode] if isinstance(mode, dict) else None

    @property
    def threshold(self) -> float:
        return float(self._config.get('Threshold', 2e5))

    @property
    def rnn_mode(self) -> str:
        mode = self._config.get('RNNMode', 'LSTM')
        return list(mode.keys())[-1] if isinstance(mode, dict) else mode

    @property
    def rnn_hidden_size(self) -> int:
        mode = self._config.get('RNNMode', 'LSTM')
        return mode[self.rnn_mode] if isinstance(mode, dict) else 64

    @property
    def input_time_step(self) -> int:
        return self._config.get('InputTimeStep', 24)

    @property
    def time_interval(self) -> int:
        return self._config.get('TimeInterval', 1)

    @property
    def predict_interval(self) -> int:
        return self._config.get('PredictInterval', 1)

    @property
    def predict_average(self) -> int:
        return self._config.get('PredictAverage', 1)


class PredictConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def start(self) -> datetime.datetime:
        info = self._config.get('Start')
        if info is None:
            raise ValueError(f'Start time not found in {self.path}')
        else:
            return datetime.datetime(**info)

    @property
    def end(self) -> datetime.datetime:
        info = self._config.get('End')
        if info is None:
            raise ValueError(f'End time not found in {self.path}')
        else:
            return datetime.datetime(**info)

    @property
    def modify_cities(self) -> dict[str, float]:
        return self._config.get('ModifyCities', {})

    def get_cities_modify_dict(self,
                               model_cities: list[str]) -> dict[str, float]:
        return {city: self.modify_cities.get(city, 1) for city in model_cities}

    @property
    def modify_pollutants(self) -> dict[str, float]:
        return self._config.get('ModifyPollutants', {})

    def get_pollutants_modify_dict(
            self, model_pollutants: list[str]) -> dict[str, float]:
        return {
            pollutant: self.modify_pollutants.get(pollutant, 1)
            for pollutant in model_pollutants
        }


class EvaluateConfig(_Config):

    def __init__(self, path: str) -> None:
        super().__init__(path)
    
    @property
    def evaluate_cities(self) -> list[str]:
        return self._config.get('EvaluateCities', [])
    
    @property
    def evaluate_targets(self) -> list[str]:
        return self._config.get('EvaluateTargets', [])