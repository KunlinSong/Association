import os

from assoc.types import *
from assoc.utils.config import ConfigHub


class Log:
    def __init__(self, dir: str) -> None:
        self.dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        for log_type in ['config', 'model', 'training', 'test', 'tensorboard']:
            log_dir = os.path.join(dir, log_type)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            setattr(self, log_type, )

    def save(self) -> None:
        self.config_hub.save(os.path.join(dir, 'config'))

    @classmethod
    def new_log(cls, dir: str) -> None:
        config_hub = ConfigHub(os.path.join(dir, 'config'))
        return cls(dir, config_hub, from_log=True)