import datetime
import os
import statistics

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

    for directory in [config_dir, log_dir, data_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # load config
    config_hub = assoc.utils.config.ConfigHub.from_config_dir(config_dir)

    # get log_hub
    log_hubs_dir_list = [
        os.path.join(log_dir, log_file) for log_file in os.listdir(log_dir)
    ]
    log_hubs_dir_list.append(
        os.path.join(log_dir, f'log_hub_{len(log_hubs_dir_list)}'))
    for log_hub_dir in log_hubs_dir_list:
        log_hub = assoc.utils.log.LogHub(log_hub_dir, config_hub)
        if log_hub.config_log.config_hub == config_hub:
            break

    # get data
    location_collection = assoc.utils.data.data.LocationCollection(
        os.path.join(data_dir, 'location.csv'), config_hub)
    dataset = assoc.utils.data.dataset.Dataset(data_dir, config_hub)

    # get dataloader
    batch_size = log_hub.config_log.config_hub.learning_config.batch_size
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
    model = log_hub.config_log.config_hub.get_model(
        location_collection.distance_matrix, dataset.time_attr_idx,
        location_collection.location_mat)
    dtype = getattr(torch, log_hub.config_log.config_hub.basic_config.dtype)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dtype)
    try:
        model.load_state_dict(log_hub.model_log.latest_state_dict)
        model.to(device)
    except FileNotFoundError:
        model.to(device)
        log_hub.tensorboard_log.add_graph(
            model,
            (train_loader.batch_size,
             log_hub.config_log.config_hub.model_config.input_time_steps,
             len(log_hub.config_log.config_hub.data_config.all_cities),
             len(log_hub.config_log.config_hub.data_config.attributes) + 4),
            dtype, device)

    loss = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=log_hub.config_log.config_hub.learning_config.learning_rate)

    start = log_hub.training_log.latest_epoch + 1
    end = log_hub.config_log.config_hub.learning_config.max_epochs
    best_epoch_info = log_hub.training_log.best_epoch_info
    best_epoch, best_train_loss, best_val_loss = best_epoch_info
    patience = log_hub.config_log.config_hub.learning_config.early_stopping_patience
    print('Start Training...')
    for epoch in range(start, end + 1):
        if epoch > 5:
            break
        if (epoch - best_epoch) > patience:
            break

        print(f'Epoch {epoch} / {end}\nAt {datetime.datetime.now()}')
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
        print(f'Train Loss: {train_loss}')

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
        log_hub.training_log.append(epoch, train_loss, val_loss)
        log_hub.tensorboard_log.append_loss(epoch, train_loss, val_loss)
        log_hub.model_log.save_latest_state_dict(model.state_dict())

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            log_hub.model_log.save_best_state_dict(model.state_dict())
        elif epoch > (best_epoch + patience):
            break

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
    true_values = true_values.transpose(2, 1, 0)
    pred_values = pred_values.transpose(2, 1, 0)
    print('Testing Finished!')
    print('Saving Results...')
    for target_idx, (true_value,
              pred_value) in enumerate(zip(true_values, pred_values)):
        target = log_hub.config_log.config_hub.data_config.targets[
            target_idx]
        for node_idx, (true_target_value, pred_target_value) in enumerate(
                zip(true_value, pred_value)):
            node = log_hub.config_log.config_hub.data_config.all_cities[
                node_idx]
            for idx, (true_node_value, pred_node_value) in enumerate(
                    zip(true_target_value, pred_target_value)):
                log_hub.tensorboard_log.append_test(idx, target, node,
                                                    true_node_value,
                                                    pred_node_value)
                log_hub.test_log[target][node].append(true_node_value,
                                                      pred_node_value)

    for target in log_hub.config_log.config_hub.data_config.targets:
        for node in log_hub.config_log.config_hub.data_config.all_cities:
            log_hub.test_log[target][node].plot().save_hexbin()
            log_hub.test_log[target][node].plot().save_plot()
        log_hub.test_log[target].plot().save_hexbin()
        log_hub.test_log[target].plot().save_plot()
    log_hub.test_log.plot().save_hexbin()
    log_hub.test_log.plot().save_plot()
    print('All Finished!')


if __name__ == '__main__':
    main()