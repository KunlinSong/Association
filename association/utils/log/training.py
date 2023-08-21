import os

import pandas as pd

from association.types import *


EPOCH_COL = 'epoch'
TRAINING_LOSS_COL = 'training_loss'
VALIDATION_LOSS_COL = 'validation_loss'


class TrainingLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.log = os.path.join(self.dirname, 'training.csv')

        if os.path.exists(self.log):
            self.df = pd.read_csv(self.log)
        else:
            self.df = pd.DataFrame(
                columns=[EPOCH_COL, TRAINING_LOSS_COL, VALIDATION_LOSS_COL])

    def append(self, epoch: int, training_loss: float,
               validation_loss: float) -> None:
        if self.df[EPOCH_COL].empty:
            self.df.loc[0] = [epoch, training_loss, validation_loss]
        else:
            if epoch != (self.latest_epoch + 1):
                raise ValueError(f'Epoch {epoch} is not the next epoch')
            else:
                self.df = pd.concat([
                    self.df,
                    pd.DataFrame(
                        [[epoch, training_loss, validation_loss]],
                        columns=[EPOCH_COL, TRAINING_LOSS_COL, VALIDATION_LOSS_COL])
                ])
        self.df.to_csv(self.log, index=False, header=True)

    @property
    def latest_epoch(self) -> int:
        return (int(self.df[EPOCH_COL].iloc[-1])
                if len(self.df[EPOCH_COL]) > 0 else 0)

    @property
    def best_epoch_info(self) -> tuple[int, float]:
        if self.df.empty:
            return (0, float('inf'))
        else:
            best_idx = self.df[VALIDATION_LOSS_COL].idxmin()
            return (self.df[EPOCH_COL].iloc[best_idx],
                    self.df[VALIDATION_LOSS_COL].iloc[best_idx])
