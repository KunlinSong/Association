import numpy as np

from association.types import *


class StatisticalMeasures:

    def __init__(self, true_val: np.ndarray, pred_val: np.ndarray) -> None:
        self.true_val = true_val
        self.pred_val = pred_val

    @property
    def mae(self) -> float:
        return np.mean(np.abs(self.true_val - self.pred_val))

    @property
    def mse(self) -> float:
        return np.mean((self.true_val - self.pred_val)**2)

    @property
    def rmse(self) -> float:
        return np.sqrt(np.mean((self.true_val - self.pred_val)**2))

    @property
    def mape(self) -> float:
        return np.mean(
            np.abs(self.true_val - self.pred_val) / self.true_val) * 100

    @property
    def r2(self) -> float:
        return (1 - np.sum((self.true_val - self.pred_val)**2) / np.sum(
            (self.true_val - np.mean(self.true_val))**2))

    @property
    def pearson(self) -> float:
        return np.corrcoef(self.true_val, self.pred_val)[0, 1]

    @property
    def ia(self) -> float:
        return (1 - np.sum((self.true_val - self.pred_val)**2) / np.sum(
            (np.abs(self.pred_val - np.mean(self.pred_val)) +
             np.abs(self.true_val - np.mean(self.true_val)))**2))