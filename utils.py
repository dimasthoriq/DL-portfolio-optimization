"""
Author: Dimas Ahmad
Description: This file contains utility functions
"""

import torch


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
    portfolio_values = torch.sum(y_pred * y_true, dim=1)
    portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

    # Compute the sharpe ratio
    return sharpe(portfolio_returns)
