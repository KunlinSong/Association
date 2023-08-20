import datetime
import os
import statistics
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from association.model import Model
from association.utils.config.confighub import ConfigHub, ConfigLog
from association.utils.data.dataset import Dataset
from association.utils.log.basic import ModelLog
from association.utils.log.training import TrainingLog
from association.utils.log.testing import TestLog
from association.types import *


def main():
    print('Preparing Data...')

    # get path
    project_dir = os.getcwd()
    config_dir = os.path.join(project_dir, 'Config')
    data_dir = os.path.join(project_dir, 'Data')
    generated_dir = os.path.join(project_dir, 'Generated')
    for directory in [config_dir, data_dir, generated_dir]:
        os.makedirs(directory, exist_ok=True)

    # load config
    config_hub = ConfigHub(config_dir)

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
    config_hub.save(os.path.join(log_dir, 'config'))

    # get data
    dataset = Dataset(data_dir, config_hub)

    # get dataloader
    dataset.to_state('training')
    train_loader = DataLoader(dataset,
                              batch_size=config_hub.batch_size,
                              shuffle=True,
                              num_workers=min(config_hub.batch_size, os.cpu_count()))
    dataset.to_state('validation')
    val_loader = DataLoader(dataset,
                            batch_size=config_hub.batch_size,
                            shuffle=False,
                            num_workers=min(config_hub.batch_size, os.cpu_count()))
    dataset.to_state('test')
    test_loader = DataLoader(dataset,
                             batch_size=config_hub.batch_size,
                             shuffle=False,
                             num_workers=min(config_hub.batch_size, os.cpu_count()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model
    model_log = ModelLog(os.path.join(log_dir, 'model'))
    model = Model(
        association_mode=config_hub.association_mode,
        association_param=config_hub.association_param,
        rnn_mode=config_hub.rnn_mode,
        rnn_hidden=config_hub.rnn_hidden_size,
        in_features=dataset.num_attributes,
        out_features=len(config_hub.targets),
        input_time_steps=config_hub.input_time_step,
        num_nodes=len(config_hub.input_cities),
        distance_matrix=dataset.distance_matrix,
        adjacency_threshold=config_hub.threshold,
        dtype=config_hub.dtype,
    )
    try:
        model.load_state_dict(model_log.latest_state_dict)
        model = model.to(device)
    except FileNotFoundError:
        model = model.to(device)

    # init basic parameters
    loss = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_hub.learning_rate)

    # get training info
    training_log = TrainingLog(os.path.join(log_dir, 'training'))
    best_epoch, best_val_loss = training_log.best_epoch_info
    print('Start Training...')
    dtype = getattr(torch, config_hub.dtype)
    for epoch in range(training_log.latest_epoch + 1, config_hub.max_epoch):
        # if epoch > 1:
        #     break
        if (epoch - best_epoch) > config_hub.early_stopping_patience:
            break

        print(f'Epoch {epoch} ({datetime.datetime.now()})')
        # train
        train_loss = []
        dataset.to_state('training')
        for source_inputs, source_targets in train_loader:
            inputs = source_inputs.to(dtype).to(device)
            targets = source_targets.to(dtype).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            model_loss = loss(outputs, targets)
            model_loss.backward()
            optimizer.step()
            train_loss.append(model_loss.item())
        train_loss = statistics.mean(train_loss)
        print(f'Training Loss: {train_loss}')

        # validation
        val_loss = []
        dataset.to_state('validation')
        with torch.no_grad():
            for source_inputs, source_targets in val_loader:
                inputs = source_inputs.to(dtype).to(device)
                targets = source_targets.to(dtype).to(device)
                outputs = model(inputs)
                model_loss = loss(outputs, targets)
                val_loss.append(model_loss.item())
        val_loss = statistics.mean(val_loss)
        print(f'Validation Loss: {val_loss}')

        # update log
        training_log.append(epoch, train_loss, val_loss)
        model_log.save_latest_state_dict(model.state_dict())

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            model_log.save_best_state_dict(model.state_dict())

    print('Training Finished!')

    # test
    print('Start Testing...')
    test_log = TestLog(
        os.path.join(log_dir, 'test'),
        config_hub.input_cities,
        config_hub.targets
    )
    true_values = []
    pred_values = []
    dataset.to_state('test')
    model.load_state_dict(model_log.best_state_dict)
    with torch.no_grad():
        for source_inputs, source_targets in test_loader:
            inputs = source_inputs.to(dtype).to(device)
            targets = source_targets.to(dtype).to(device)
            outputs = model(inputs)
            true_values.append(targets[:, -1, :, :].cpu().numpy())
            pred_values.append(outputs[:, -1, :, :].cpu().numpy())
    true_values = np.concatenate(true_values, axis=0)
    pred_values = np.concatenate(pred_values, axis=0)
    true_values = true_values.transpose(2, 1, 0)
    pred_values = pred_values.transpose(2, 1, 0)
    print('Testing Finished!')
    print('Saving Results...')
    for target_idx, (true_value,
                     pred_value) in enumerate(zip(true_values, pred_values)):
        target = config_hub.targets[target_idx]
        for node_idx, (true_target_value, pred_target_value) in enumerate(
                zip(true_value, pred_value)):
            node = config_hub.input_cities[
                node_idx]
            for _, (true_node_value, pred_node_value) in enumerate(
                    zip(true_target_value, pred_target_value)):
                test_log[target][node].append(true_node_value,
                                              pred_node_value)
    print('All Finished!')


if __name__ == '__main__':
    main()
