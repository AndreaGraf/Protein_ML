import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.special import betaincinv, erfinv
from yaml import load




class GaussNLLLoss(nn.Module):
    """
    Gaussian negative log likelihood loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nll = nn.GaussianNLLLoss(eps=0.000001, full=False, reduction="mean")

    def forward(self, yhat, ytrue):
        var = torch.ones(yhat.shape, requires_grad=True, device=yhat.device)
        return self.nll(yhat, ytrue, var)
    
class GaussNLL_VAR_Loss(nn.Module):
    """
    Gaussian negative log likelihood loss.
    """

    def __init__(self) -> None:
        super().__init__()
        #make the variance a trainable parameter
        self.var = torch.nn.Parameter(torch.ones(1))
        self.nll = nn.GaussianNLLLoss(eps=0.000001, full=False, reduction="mean")

    def forward(self, yhat, ytrue):
        var = self.var.repeat(yhat.shape[0])
        #var = torch.ones(yhat.shape, requires_grad=True, device=yhat.device)
        return self.nll(yhat, ytrue, var)


# The Variational Autoencoder Loss Functions that do not require decoder structure
def KLD_diag_gaussians( mu, logvar, p_mu, p_logvar):
    """
    KL divergence between diagonal gaussian with prior diagonal gaussian.
    """
    KLD = (
            0.5 * (p_logvar - logvar)
            + 0.5 * (torch.exp(logvar) + torch.pow(mu - p_mu, 2)) / (torch.exp(p_logvar) + 1e-20)
            - 0.5
        )

    return torch.sum(KLD)


  

    
