"""Module containing loss functions for training,
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F
from yaml import load


class MSE_Loss(nn.MSELoss):
    """wrapper for a simple MSE loss with the same return shape. Forward function takes
        Args: 
            yhat, ytrue
        Returns:    
            MSE loss mean
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, yhat, ytrue):
        return self.mse(yhat, ytrue)


class GaussNLLLoss(nn.Module):
    """
    Wrapper for a Gaussian negative log likelihood loss. Forward: 
        Args:
            yhat, ytrue
        Returns:
            NLL loss mean
    """

    def __init__(self) -> None:
        super().__init__()
        self.nll = nn.GaussianNLLLoss(eps=0.000001, full=False, reduction="mean")

    def forward(self, yhat, ytrue):
        var = torch.ones(yhat.shape, requires_grad=True, device=yhat.device)
        return self.nll(yhat, ytrue, var)
    


class GaussNLL_VAR0_Loss(nn.Module):
    """
    Gaussian negative log likelihood loss with a trainable variance parameter. Forward:
        Args:
            yhat, ytrue 
        Returns:
            NLL loss mean
    """

    def __init__(self) -> None:
        super().__init__()
        #make the variance a trainable parameter
        self.var = torch.nn.Parameter(torch.ones(1))
        nn.init.trunc_normal_(self.var)
        self.nll = nn.GaussianNLLLoss(eps=0.000001, full=True, reduction="mean")

    def forward(self, yhat, ytrue):
        var = self.var.repeat(yhat.shape[0])
        return self.nll(yhat, ytrue, var)


class GaussNLL_VAR_Loss(nn.Module):
    """
        Gaussian negative log likelihood loss treating logsigma as a second order polynomial expansion 
        similar to the noisemodel in MAVE NN. Forward:
        Args:
            yhat, ytrue 
        Returns:
            NLL loss mean
    """

    def __init__(self) -> None:
        super(GaussNLL_VAR_Loss, self).__init__()

        self.a = torch.nn.Parameter(torch.empty(3,1))
        nn.init.trunc_normal_(self.a)
        self.nll = nn.GaussianNLLLoss(eps=0.000001, full=True, reduction="mean")

    
    def calc_logsigma(self, yhat):
        """ function to compute the variance based on a order polinomial expansion"""

        a = self.a.repeat(yhat.shape[0], 1, 1)      
        logsigma = a[:, 0, :] + a[:,1,:] * yhat + a[:,2,:] * torch.pow(yhat,2)

        return logsigma
    

    def forward(self, yhat, ytrue):
        var = self.calc_logsigma(yhat)
        var = torch.clamp(var, min=-20., max= 20.)
        sigma = torch.exp(var)
        nlls = 0.5 * torch.square((ytrue - yhat)) / sigma + var + 0.5 * np.log(2 * np.pi)
        return nlls.mean()

    

class NoiseLayer(nn.Module):
    """Base class for original MAVE-NN noise layers
       Args:
            model_params (dict): Dictionary of model parameters.
            key: polynomial_order

        Attributes:
            poly_order (int): Order of polynomial expansion for noise model.
    """
    def __init__(self, model_params):
        super().__init__()
        self.poly_order = model_params["polynomial_order"]
        
    @abstractmethod
    def compute_nlls(self, yhat, ytrue):
        """
        Compute the negative log-likelihoods for the given predictions and targets
        defined in the derived noise model classes.

        Args:
            yhat (torch.Tensor): Predictions from the model.
            ytrue (torch.Tensor): Targets for the model.

        Returns:
            torch.Tensor: The negative log-likelihoods for each sample in the batch.
        """
        pass

    def forward(self, yhat, ytrue):
        nlls = self.compute_nlls(yhat, ytrue)
        return nlls


class GaussianNoise(NoiseLayer):
    """A Gaussian noise distribution for GE regression 
       Args:    
            model_params (dict): Dictionary of model parameters.

        Attributes:
            poly_order (int): Order of polynomial expansion for noise model.

        Methods:
            compute_params: Compute the value of logsigma from a polynomial expansion 
            compute_nlls: Compute the negative log likelihood using the computed logsigma
        """

    def __init__(self, model_params):
        """Construct layer instance."""
        super().__init__(model_params)
        self.a = torch.nn.Parameter(torch.empty(self.poly_order + 1, 1))
        nn.init.trunc_normal_(self.a)

    def compute_params(self, yhat, y_true=None):
        """Compute layer parameters governing p(y|yhat)."""
        # Have to treat 0'th order term separately because of NaN bug.
        a = self.a.repeat(yhat.shape[0], 1, 1)
        logsigma = a[:, 0, :]

        # Add higher order terms and return
        for k in range(1, self.poly_order + 1):
            logsigma = logsigma + a[:, k, :] * torch.pow(yhat, k)
        return logsigma

    def compute_nlls(self, yhat, ytrue):
        """Compute negative log likelihood contributions for each datum."""

        logsigma = self.compute_params(yhat)
        logsigma = torch.clamp(logsigma, min=-20., max= 20.)
        sigma = torch.exp(logsigma)
        nlls = 0.5 * torch.square((ytrue - yhat) / sigma) + logsigma + 0.5 * np.log(2 * np.pi)
        return nlls


# The Variational Autoencoder Loss Functions that do not require decoder structure
def KLD_diag_gaussians( mu:torch.Tensor, logvar:torch.Tensor, p_mu:torch.Tensor, p_logvar:torch.Tensor)->torch.Tensor:
    """
    KL divergence between diagonal gaussian with prior diagonal gaussian.
    Args:
        mu (torch.Tensor): mean of the posterior
        logvar (torch.Tensor): log variance of the posterior
        p_mu (torch.Tensor): mean of the prior
        p_logvar (torch.Tensor): log variance of the prior
    Returns:
        KL divergence (torch.Tensor)
    """
    KLD = (
            0.5 * (p_logvar - logvar)
            + 0.5 * (torch.exp(logvar) + torch.pow(mu - p_mu, 2)) / (torch.exp(p_logvar) + 1e-20)
            - 0.5
        )

    return torch.sum(KLD)


  

    
