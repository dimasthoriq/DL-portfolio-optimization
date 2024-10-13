"""
Author: Dimas Ahmad
Description: This file is used to train the model
"""

import os
import time
import datetime

from losses import NegativeSharpeLoss
from utils import *


def train_one_epoch(model, optimizer, criterion, data_loader, config):
    epoch_loss = 0
    model.train()

    for i, (x, y) in enumerate(data_loader):
        x, y = x.float().to(config['device']), y.float().to(config['device'])
        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().cpu().item()
    epoch_loss /= i+1
    return epoch_loss


def validate(model, criterion, data_loader, config):
    epoch_loss = 0
    model.eval()

    for i, (x, y) in enumerate(data_loader):
        x, y = x.float().to(config['device']), y.float().to(config['device'])
        y_pred = model(x)

        loss = criterion(y_pred, y)
        epoch_loss += loss.detach().cpu().item()
    epoch_loss /= i+1
    return epoch_loss


def training(model, train_loader, val_loader, config):
    # Initialize model, loss function, and optimizer
    model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'],
                                 )
    criterion = NegativeSharpeLoss()

    # Initialize the variables to store the best states
    best_model_state = model.state_dict()
    best_loss = torch.tensor(float('inf'))
    best_epoch = 0

    # Initialize the variables to store the losses
    loss_train = []
    loss_val = []

    # Train the model
    time_start = time.time()
    for epoch in range(config['epochs']):
        # Train the model
        epoch_loss = train_one_epoch(model, optimizer, criterion, train_loader, config)
        epoch_val_loss = validate(model, criterion, val_loader, config)

        # Store the losses
        loss_train.append(epoch_loss)
        loss_val.append(epoch_val_loss)

        # Print the losses
        print(f"Epoch: {epoch+1}/{config['epochs']} | Train Sharpe: {-epoch_loss:.4f} | Val Sharpe: {-epoch_val_loss:.4f}")

        if epoch_val_loss < best_loss:
            best_model_state = model.state_dict()
            best_loss = epoch_val_loss
            best_epoch = epoch
            print(f"Best model updated at epoch {epoch+1}")

    duration = time.time() - time_start
    time_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    print('Training completed in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    # Load the best model
    model.load_state_dict(best_model_state)

    # Define the path to save the model
    experiment_path = config['experiment_path']
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    model_save_path = os.path.join(experiment_path,
                                   f'DLS_vol{config['volatility_scaling']}_C{config['cost_rate']}_{time_stamp}.pth'
                                   )
    torch.save(model, model_save_path)
    return model, loss_train, loss_val, best_epoch
