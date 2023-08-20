import os

import torch

from association.types import *


class ModelLog:

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        os.makedirs(self.dirname, exist_ok=True)
        self.latest_path = os.path.join(self.dirname, 'latest.pth')
        self.best_path = os.path.join(self.dirname, 'best.pth')

    @property
    def best_state_dict(self) -> Any:
        if os.path.exists(self.best_path):
            return torch.load(self.best_path)
        else:
            raise FileNotFoundError(f'Best model not found in {self.dirname}')

    @property
    def latest_state_dict(self) -> Any:
        if os.path.exists(self.latest_path):
            return torch.load(self.latest_path)
        else:
            raise FileNotFoundError(
                f'Latest model not found in {self.dirname}')

    def save_best_state_dict(self, state_dict: Any) -> None:
        torch.save(state_dict, self.best_path)

    def save_latest_state_dict(self, state_dict: Any) -> None:
        torch.save(state_dict, self.latest_path)
