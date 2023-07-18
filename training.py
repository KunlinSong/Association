import datetime
import os
import statistics
from contextlib import suppress

import numpy as np
import torch
from torch.utils.data import DataLoader

import assoc
from assoc.types import *


def main():
    print('Preparing Data...')

    # get path
    project_dir = os.getcwd()
    config_dir = os.path.join(project_dir, 'Config')
    log_dir = os.path.join(project_dir, 'Log')
    data_dir = os.path.join(project_dir, 'Data')

    # load config
    config_hub = assoc.utils.config.ConfigHub()
    config_hub.load(config_dir)

    # get log_hub
    log_hubs_dir_list = [
        os.path.join(log_dir, log_file) for log_file in os.listdir(log_dir)
    ]
    for log_hub_dir in log_hubs_dir_list:
        log_hub = assoc.utils.log.LogHub(log_hub_dir)
        if log_hub.config_log == config_hub:
            break
    else:
        log_hub_dir = os.path.join(log_dir,
                                   f'log_hub_{len(log_hubs_dir_list)}')
        log_hub = assoc.utils.log.LogHub.from_config_hub(
            log_hub_dir, config_hub)

    # get data
    time_dict = assoc.utils.data.data.TimeDict(data_dir, config_hub)
    location_collection = assoc.utils.data.data.LocationCollection(
        os.path.join(data_dir, 'location.csv'), config_hub)
    dataset = assoc.utils.data.dataset.Dataset(data_dir, config_hub)

    # get dataloader
    batch_size = config_hub.learning_config.batch_size
    dataset.to_state('training')
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=min(batch_size, os.cpu_count()))
    dataset.to_state('validation')
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=min(batch_size, os.cpu_count()))
    dataset.to_state('test')
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=min(batch_size, os.cpu_count()))

    # get model
    model = config_hub.get_model(location_collection.distance_matrix,
                                 dataset.time_attr_idx,
                                 location_collection.location_mat)
    try:
        state_dict = log_hub.model_log.load_latest_state_dict()
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        log_hub.tensorboard_log.add_graph(
            model,
            (train_loader.batch_size, config_hub.model_config.input_time_steps,
             len(config_hub.data_config.all_cities),
             len(config_hub.data_config.attributes) + 4))

    dtype = getattr(torch, config_hub.basic_config.dtype)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(dtype)
    model = model.to(device)
    loss = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config_hub.learning_config.learning_rate)

    start = log_hub.training_log.latest_epoch + 1
    end = config_hub.learning_config.max_epochs
    best_epoch_info = log_hub.training_log.best_epoch_info
    best_epoch, best_train_loss, best_val_loss = best_epoch_info
    patience = config_hub.learning_config.early_stopping_patience
    print('Start Training...')
    for epoch in range(start, end):
        if (epoch - best_epoch) > patience:
            break

        print(f'Epoch {epoch + 1} / {end}\nAt {datetime.datetime.now()}')
        # train
        train_loss = []
        dataset.to_state('training')
        for source_inputs, source_targets in train_loader:
            inputs = source_inputs.to(dtype).to(device)
            targets = source_targets.to(dtype).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = statistics.mean(train_loss)
        print(f'Train Loss: {train_loss}')

        # validation
        val_loss = []
        dataset.to_state('validation')
        with torch.no_grad():
            for source_inputs, source_targets in val_loader:
                inputs = source_inputs.to(dtype).to(device)
                targets = source_targets.to(dtype).to(device)
                outputs = model(inputs)
                loss = loss(outputs, targets)
                val_loss.append(loss.item())
        val_loss = statistics.mean(val_loss)
        print(f'Validation Loss: {val_loss}')

        # update log
        log_hub.training_log.append(epoch, train_loss, val_loss)
        log_hub.tensorboard_log.append_loss(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            log_hub.model_log.save_best_state_dict(model.state_dict())
        log_hub.model_log.save_latest_state_dict(model.state_dict())

    print('Training Finished!')

    # test
    print('Start Testing...')
    true_values = []
    pred_values = []
    dataset.to_state('test')
    with torch.no_grad():
        for source_inputs, source_targets in test_loader:
            inputs = source_inputs.to(dtype).to(device)
            targets = source_targets.to(dtype).to(device)
            outputs = model(inputs)
            true_values.append(targets[:, -1, :, :].cpu().numpy())
            pred_values.append(outputs[:, -1, :, :].cpu().numpy())
    true_values = np.concatenate(true_values, axis=0)
    pred_values = np.concatenate(pred_values, axis=0)
    print('Testing Finished!')
    print('Saving Results...')
    for idx, (true_value,
              pred_value) in enumerate(zip(true_values, pred_values)):
        for node_idx, (true_node_value, pred_node_value) in enumerate(
                zip(true_value, pred_value)):
            node = config_hub.data_config.all_cities[node_idx]
            for target_idx, (true_target_value,
                             pred_target_value) in enumerate(
                                 zip(true_node_value, pred_node_value)):
                target = config_hub.data_config.attributes[target_idx]
                log_hub.tensorboard_log.append_test(idx, target, node,
                                                    true_target_value,
                                                    pred_target_value)
                log_hub.test_log[target][node].append(true_target_value,
                                                      pred_target_value)
    for node in config_hub.data_config.all_cities:
        for target in config_hub.data_config.attributes:
            log_hub.test_log[target][node].plot().save_hexbin()
            log_hub.test_log[target][node].plot().save_plot()
        log_hub.test_log[target].plot().save_hexbin()
        log_hub.test_log[target].plot().save_plot()
    log_hub.test_log.plot().save_hexbin()
    log_hub.test_log.plot().save_plot()
    print('All Finished!')

if __name__ == '__main__':
    main()