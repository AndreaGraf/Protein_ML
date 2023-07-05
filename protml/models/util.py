"""Utility functions used by different models"""

from typing import List

import torch
import torch.nn as nn
from omegaconf import open_dict

nonlinearities_map = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": nn.functional.elu,
    "linear": nn.Identity,
}


def get_l2_norm(model: torch.nn.Module) -> torch.Tensor:
    """
    Compute the L2 norm of the module weights.
    Args:
        model: pytorch module - the module to calculate the l2 norm for
    """
    params = torch.cat([x.view(-1) for x in model.parameters()])
    return torch.norm(params)


def sample_latent(mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
    """
    Samples a latent vector via reparametrization trick
    Args:
        mu(torch.Tensor): mean of the latent distribution
        log_var(torch.Tensor): log variance of the latent distribution
    Returns:
        z(torch.Tensor): latent vector
    """
    eps = torch.randn_like(mu)
    z = torch.exp(0.5 * log_var) * eps + mu
    return z



