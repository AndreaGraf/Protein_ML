""" Module defining the measurement maps from the encoded sequence to a single value prediction
"""

import torch
import torch.nn as nn


class Mave_Global_Epistasis_Measurement(nn.Module):
    """
    Nonlinear GE Measurement process. for z_dim = 1 this is the same as in MAVE-NN
        Args:
            model_params (dict): Dictionary of model parameters
            keys: num_hidden_nodes, z_dim                    

        Attributes:
            num_hidden_nodes (int): Number of hidden nodes in the measurement process.
            z_dim (int): Dimension of the latent space.
    """

    def __init__(self, model_params):

        super().__init__()

        self.num_hidden_nodes = model_params["num_hidden_nodes"]
        self.z_dim = model_params["z_dim"]
        self.a_0 = torch.nn.Parameter(torch.randn(1))
        self.bk = torch.nn.Parameter(torch.randn(self.num_hidden_nodes))
        self.ck = torch.nn.Parameter(torch.randn((self.num_hidden_nodes, self.z_dim)))
        self.dk = torch.nn.Parameter(torch.randn(self.num_hidden_nodes))

    
    def forward(self, z):
        """Compute y_hat from sample of z"""
        yh = [self.a_0 + torch.sum(self.bk * torch.tanh(self.ck @ zi + self.dk)) for zi in z]

        return torch.stack(yh).view(z.shape[0], 1)


class Linear_Measurement(nn.Module):
    """
    Linear Measurement head mapping z to yhat
        Args:
            model_params (dict): Dictionary of model parameters
            keys: z_dim, mu_bias_init(optional)                    

        Attributes:
            num_hidden_nodes (int): Number of hidden nodes in the measurement process.
            z_dim (int): Dimension of the latent space.
    """

    def __init__(self, model_params):

        super().__init__()
        self.z_dim = model_params["z_dim"]
        self.mu_bias_init = model_params.get("mu_bias_init", 0.1)
        self.mp_lin = nn.Linear(self.z_dim, 1)
        nn.init.constant_(self.mp_lin.bias, self.mu_bias_init)

    def forward(self, yh):
        return self.mp_lin(yh)
