"""
Author: Dimas Ahmad
Description: This file contains the negative sharpe ratio as described by the original paper.
"""

import torch

from utils import sharpe


class NegativeSharpeLoss(torch.nn.Module):
    def __init__(self):
        super(NegativeSharpeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the negative sharpe ratio of the returns.
        """
        # Ensure consistency in the dtype of the inputs
        if y_pred.dtype != y_true.dtype:
            y_pred = y_pred.to(y_true.dtype)

        # Compute the portfolio values and returns
        portfolio_values = torch.sum(y_pred * y_true, dim=1)
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

        # Compute the sharpe ratio
        return -sharpe(portfolio_returns)
