import os
from glob import glob

import numpy as np
import pandas as pd

from association.utils.config.confighub import EvaluateHub, ConfigLog
from association.utils.evaluate import StatisticalMeasures
from association.utils.log.testing import TestLog
from association.types import *


def main():
    print('Evaluating...')

    # get path
    project_dir = os.getcwd()
    config_dir = os.path.join(project_dir, 'Config')
    generated_dir = os.path.join(project_dir, 'Generated')

    # load config
    config_hub = EvaluateHub(config_dir)

    # compare config log
    logs_dir_list = glob(os.path.join(generated_dir, 'log_*'))
    logs_dir_list.append(
        os.path.join(generated_dir, f'log_hub_{len(logs_dir_list)}'))
    for log_dir in logs_dir_list:
        try:
            config_log = ConfigLog(os.path.join(log_dir, 'config'))
            if config_log == config_hub:
                break
        except FileNotFoundError:
            break
    config_hub.save(os.path.join(log_dir, 'evaluate'))

    test_log = TestLog(
        os.path.join(log_dir, 'test'),
        config_hub.input_cities,
        config_hub.targets
    )

    true_vals = []
    pred_vals = []
    for target in config_hub.evaluate_targets:
        for node in config_hub.evaluate_cities:
            true_vals.append(test_log[target][node].true_val)
            pred_vals.append(test_log[target][node].pred_val)
    true_vals = np.stack(true_vals, axis=0)
    pred_vals = np.stack(pred_vals, axis=0)

    result = StatisticalMeasures(true_val=true_vals, pred_val=pred_vals)
    df = pd.DataFrame(
        {
            'mae': [result.mae],
            'mse': [result.mse],
            'rmse': [result.rmse],
            'mape': [result.mape],
            'r2': [result.r2],
            'pearson': [result.pearson],
            'ia': [result.ia],
        }
    )
    df.to_csv(os.path.join(log_dir, 'evaluate', 'evaluate.csv'), index=False)
    print('Finish.')


if __name__ == '__main__':
    main()