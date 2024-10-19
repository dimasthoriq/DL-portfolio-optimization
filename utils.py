"""
Author: Dimas Ahmad
Description: This file contains utility functions
"""
import pandas as pd
import numpy as np
import torch
import yfinance


def sharpe(returns):
    """
    Compute the Sharpe ratio of the returns.
    """
    return torch.mean(returns) / torch.std(returns)


def evaluate_sharpe(y_pred, y_true):
    """
    Compute the sharpe ratio of the predicted portfolio weights.
    """
    # Ensure consistency in the dtype of the inputs
    if y_pred.dtype != y_true.dtype:
        y_pred = y_pred.to(y_true.dtype)

    # Compute the portfolio values and returns
    portfolio_returns = torch.sum(y_pred * y_true, dim=1)
    # portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

    # Compute the sharpe ratio
    return sharpe(portfolio_returns)


def get_data(etfs, start_date='2006-02-06', end_date='2020-01-01'):
    data = pd.DataFrame()
    for etf in etfs:
        data[etf] = yfinance.Ticker(etf).history(start=start_date, end=end_date)['Close'].reset_index(drop=True)

    dates = yfinance.Ticker('AGG').history(start=start_date, end=end_date).index
    data.index = dates
    return data


def get_returns(data):
    returns = data.pct_change()
    returns.fillna(0, inplace=True)
    returns.columns = ['r_VTI', 'r_AGG', 'r_DBC', 'r_VIX']
    returns.index = data.index
    return pd.concat([data, returns], axis=1)


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_data = data[:int(train_ratio*len(data))]
    val_data = data[int(train_ratio*len(data)):int((train_ratio+val_ratio)*len(data))]
    test_data = data[-int(test_ratio*len(data)):]
    return train_data, val_data, test_data


def standardize(data, scaler=None):
    std_data = data.copy()
    if scaler is None:
        scaler = {}
        for key in data.columns:
            scaler[key] = {'mean': data[key].mean(), 'std': data[key].std()}
            std_data.loc[:, key] = (std_data.loc[:, key] - scaler[key]['mean']) / scaler[key]['std']
        return std_data, scaler
    else:
        for key in data.columns:
            std_data.loc[:, key] = (std_data.loc[:, key] - scaler[key]['mean']) / scaler[key]['std']
        return std_data


def get_sequences(scaled_data, data, sequence_length=50):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(scaled_data[i:i + sequence_length].values)
        y.append(data.iloc[i + sequence_length].values)

    x = np.array(x)
    y = np.array(y)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_loaders(x, y, batch_size=64, shuffle=True, num_workers=4):
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def get_tensors(loader):
    x_tensors = []
    y_tensors = []
    for _, (X, y) in enumerate(loader):
        x_tensors.append(X)
        y_tensors.append(y)
    return torch.cat(x_tensors), torch.cat(y_tensors)


def data_preprocessing(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seq_len=50, batch_size=64, shuffle_train=False, num_workers=4):
    data_with_returns = get_returns(data)
    train, val, test = split_data(data_with_returns, train_ratio, val_ratio, test_ratio)

    scaled_train, scaler = standardize(train)
    scaled_val = standardize(val, scaler)
    scaled_test = standardize(test, scaler)

    x_train, y_train = get_sequences(scaled_train, train.loc[:, ['r_VTI', 'r_AGG', 'r_DBC', 'r_VIX']], seq_len)
    x_val, y_val = get_sequences(scaled_val, val.loc[:, ['r_VTI', 'r_AGG', 'r_DBC', 'r_VIX']], seq_len)
    x_test, y_test = get_sequences(scaled_test, test.loc[:, ['r_VTI', 'r_AGG', 'r_DBC', 'r_VIX']], seq_len)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    train_loader = get_loaders(x_train, y_train, batch_size, shuffle_train, num_workers)
    val_loader = get_loaders(x_val, y_val, batch_size, False, num_workers)
    test_loader = get_loaders(x_test, y_test, batch_size, False, num_workers)

    return train_loader, val_loader, test_loader, scaler
