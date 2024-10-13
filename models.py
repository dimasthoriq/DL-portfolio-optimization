"""
Author: Dimas Ahmad
Description: This file contains the pytorch implementation of the models described in the original paper
"""

import torch


class DLS(torch.nn.Module):
    """
    Description: This class defines the proposed LSTM network with a single layer of 64 hidden units
    Input: A tensor of shape (batch_size, seq_len, input_dim)
    Output: A tensor of shape (batch_size, seq_len, output_dim) representing the portfolio weights

    Where output_dim is the number of assets in the portfolio (in this case 4 since the paper uses 4 asset classes idx),
    and input_dim is 2*output_dim since the input is the concatenation of the price and return of the assets.

    Uses softmax activation in the output layer.
    """
    def __init__(self, config):
        super(DLS, self).__init__()
        # Parse the parameters from configuration
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.input_dim = 2*self.output_dim
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        # Build the network
        self.lstm = torch.nn.LSTM(self.input_dim,
                                  self.hidden_dim,
                                  self.num_layers,
                                  dropout=self.dropout,
                                  batch_first=True
                                  )

        self.fc = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Forward pass
        lstm_out, _ = self.lstm(x)
        # Get the output from the last time step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        # Apply softmax activation
        return torch.nn.functional.softmax(out, dim=1)
