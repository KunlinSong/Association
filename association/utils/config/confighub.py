import datetime
import os

from association.types import *
from association.utils.config.config import *


class ConfigLog:
    DATA_CONFIG_BASENAME = 'Data.yaml'
    MODEL_CONFIG_BASENAME = 'Model.yaml'

    def __init__(self, dirname: str) -> None:
        self._data_config = DataConfig(
            os.path.join(dirname, self.DATA_CONFIG_BASENAME))
        self._model_config = ModelConfig(
            os.path.join(dirname, self.MODEL_CONFIG_BASENAME))

    def __eq__(self, other: 'ConfigLog') -> bool:
        return ((self._data_config == other._data_config)
                and (self._model_config == other._model_config))

    def save(self, dirname: str) -> None:
        os.makedirs(dirname, exist_ok=True)
        self._data_config.save(os.path.join(dirname,
                                            self.DATA_CONFIG_BASENAME))
        self._model_config.save(
            os.path.join(dirname, self.MODEL_CONFIG_BASENAME))


class ConfigHub(ConfigLog):
    BASIC_CONFIG_BASENAME = 'Basic.yaml'
    LEARNING_CONFIG_BASENAME = 'Learning.yaml'
    POLLUTANTS = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']

    def __init__(self, dirname: str) -> None:
        super().__init__(dirname)
        self._basic_config = BasicConfig(
            os.path.join(dirname, self.BASIC_CONFIG_BASENAME))
        self._learning_config = LearningConfig(
            os.path.join(dirname, self.LEARNING_CONFIG_BASENAME))

    def strptime(self, time_str: str) -> datetime.datetime:
        return self._basic_config.strptime(time_str=time_str)

    def strftime(self, time: datetime.datetime) -> str:
        return self._basic_config.strftime(time=time)

    @property
    def time_attribute_name(self) -> str:
        return self._basic_config.time_attribute_name

    @property
    def longitude_attribute_name(self) -> str:
        return self._basic_config.longitude_attribute_name

    @property
    def latitude_attribute_name(self) -> str:
        return self._basic_config.latitude_attribute_name

    @property
    def elevation_attribute_name(self) -> str:
        return self._basic_config.elevation_attribute_name

    @property
    def dtype(self) -> str:
        return self._basic_config.dtype

    @property
    def input_cities(self) -> list[str]:
        return self._data_config.input_cities

    @property
    def input_attributes(self) -> list[str]:
        return self._data_config.input_attributes

    @property
    def targets(self) -> list[str]:
        return self._data_config.targets

    def get_training_size(self, total_size: int) -> int:
        return self._data_config.get_training_size(total_size=total_size)

    def get_validation_size(self, total_size: int) -> int:
        return self._data_config.get_validation_size(total_size=total_size)

    def get_test_size(self, total_size: int) -> int:
        return self._data_config.get_test_size(total_size=total_size)

    @property
    def batch_size(self) -> int:
        return self._learning_config.batch_size

    @property
    def learning_rate(self) -> float:
        return self._learning_config.learning_rate

    @property
    def weight_decay(self) -> float:
        return self._learning_config.weight_decay

    @property
    def max_epoch(self) -> int:
        return self._learning_config.max_epoch

    @property
    def early_stopping_patience(self) -> int:
        return self._learning_config.early_stopping_patience

    @property
    def association_mode(self) -> str:
        return self._model_config.association_mode

    @property
    def association_param(self) -> int:
        return self._model_config.association_param

    @property
    def threshold(self) -> float:
        return self._model_config.threshold

    @property
    def rnn_mode(self) -> str:
        return self._model_config.rnn_mode

    @property
    def rnn_hidden_size(self) -> int:
        return self._model_config.rnn_hidden_size

    @property
    def input_time_step(self) -> int:
        return self._model_config.input_time_step

    @property
    def time_interval(self) -> int:
        return self._model_config.time_interval

    @property
    def predict_interval(self) -> int:
        return self._model_config.predict_interval

    @property
    def predict_average(self) -> bool:
        return self._model_config.predict_average


class PredictHub(ConfigHub):
    PREDICT_CONFIG_BASENAME = 'Predict.yaml'
    def __init__(self, dirname: str) -> None:
        super().__init__(dirname)
        self._predict_config = PredictConfig(
            os.path.join(dirname, self.PREDICT_CONFIG_BASENAME))
    
    @property
    def cities_modify_dict(self) -> dict[str, float]:
        if self._predict_config is None:
            raise AttributeError('cities_modify_dict is not available. \
                                 Please check if this is the right \
                                 ConfigHub instance.')
        else:
            return self._predict_config.get_cities_modify_dict(
                model_cities=self.input_cities)

    @property
    def pollutants_modify_dict(self) -> dict[str, float]:
        if self._predict_config is None:
            raise AttributeError('pollutants_modify_dict is not available. \
                                 Please check if this is the right \
                                 ConfigHub instance.')
        else:
            return self._predict_config.get_pollutants_modify_dict(
                model_pollutants=self.targets)

    @property
    def range_datetimes(self) -> tuple[datetime.datetime, datetime.datetime]:
        start = self._predict_config.start - (
            self.predict_interval + self.input_time_step -
            1) * datetime.timedelta(hours=self.time_interval)
        end = self._predict_config.end + (self.predict_average -
                                          1) * datetime.timedelta(
                                              hours=self.time_interval)
        return start, end

    def save(self, dirname: str) -> None:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._predict_config.save(os.path.join(dirname, self.PREDICT_CONFIG_BASENAME))


class AssociationMatrixHub(ConfigHub):
    ASSOCIATION_MATRIX_CONFIG_BASENAME = 'AssociationMatrix.yaml'

    def __init__(self, dirname: str) -> None:
        super().__init__(dirname)
        self._association_matrix_config = AssociationMatrixConfig(
            os.path.join(dirname, self.ASSOCIATION_MATRIX_CONFIG_BASENAME))
        
    @property
    def segment_datetimes(self) -> Optional[list[str]]:
        if self._association_matrix_config is None:
            raise AttributeError('segment_datetimes is not available. \
                                 Please check if this is the right ConfigHub \
                                 instance.')
        else:
            return [
                self.strftime(time)
                for time in self._association_matrix_config.segment_datetimes
            ]
    
    def save(self, dirname: str) -> None:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._association_matrix_config.save(os.path.join(dirname, self.ASSOCIATION_MATRIX_CONFIG_BASENAME))


class EvaluateHub(ConfigHub):
    PREDICT_CONFIG_BASENAME = 'Evaluate.yaml'
    def __init__(self, dirname: str) -> None:
        super().__init__(dirname)
        self._evaluate_config = EvaluateConfig(
            os.path.join(dirname, self.PREDICT_CONFIG_BASENAME))

    @property
    def evaluate_cities(self) -> list[str]:
        return self._evaluate_config.evaluate_cities
    
    @property
    def evaluate_targets(self) -> list[str]:
        return self._evaluate_config.evaluate_targets

    def save(self, dirname: str) -> None:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._evaluate_config.save(os.path.join(dirname, self.PREDICT_CONFIG_BASENAME))