import numpy as np
import torch


def xavier_init_routine(m):
    """function for weight initialization"""
    if type(m) == torch.nn.Linear:

        # manual xavier initialization
        xavier_stddev = np.sqrt(2 / (m.in_features + m.out_features))
        torch.nn.init.trunc_normal_(m.weight, mean=0, std=xavier_stddev)
        # torch.nn.init.xavier_normal_(m.weight, gain=1)

        m.bias.data.fill_(0)


def weighted_mse_loss(input, target):
    weights = np.ones(len(input))
    weights[0:10] = 10
    weights = torch.tensor(weights, dtype=torch.float64)

    return torch.sum(weights * (input - target) ** 2)
