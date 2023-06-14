import torch
import torch.nn as nn


class Mave_Global_Epistasis_Measurement(nn.Module):
    """
    Nonlinear GE Measurement of latent space from MAVE-NN
    """

    def __init__(self, model_params):
        """ """
        super().__init__()

        self.num_hidden_nodes = model_params["num_hidden_nodes"]
        self.z_dim = model_params["z_dim"]
        self.a_0 = torch.nn.Parameter(torch.randn(1))
        self.bk = torch.nn.Parameter(torch.randn(self.num_hidden_nodes))
        self.ck = torch.nn.Parameter(torch.randn((self.num_hidden_nodes, self.z_dim)))
        self.dk = torch.nn.Parameter(torch.randn(self.num_hidden_nodes))

    
    # TODO - find a way to use matmult layers
    def forward(self, z):
        """Compute y_hat from sample of z"""
        yh = []
        for zi in z:
            yh.append(self.a_0 + torch.sum(self.bk * torch.tanh(self.ck @ zi + self.dk)))

        return torch.stack(yh).view(z.shape[0], 1)



class Linear_Measurement(nn.Module):
    """linear measurement from the latent 'phenotype'"""

    def __init__(self, model_params):
        """
        Required input parameters:
        z_dim: dimension of latent space
        """
        super().__init__()
        self.z_dim = model_params["z_dim"]
        self.mu_bias_init = model_params.get("mu_bias_init", 0.1)
        self.mp_lin = nn.Linear(self.z_dim, 1)
        nn.init.constant_(self.mp_lin.bias, self.mu_bias_init)

    def forward(self, yh):
        return self.mp_lin(yh)
