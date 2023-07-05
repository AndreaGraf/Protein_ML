from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
#from scipy.special import erfinv

from ..components.loss_functions import KLD_diag_gaussians
from ..util import nonlinearities_map


class MLPDecoder(nn.Module):
    """
    standard multilayer perceptron decoder class
    Args:
        model_params: dictionary of model parameters
        dataset_params: reqiuired information on the dataset 
    Attributes:
        seq_len: length of the sequence
        alphabet_size: size of the alphabet
        hidden_layer_sizes: list of hidden layer sizes
        z_dim: dimension of the latent space
        dropout_prob: dropout probability
        include_temperature_scaler: whether to include a temperature scaler
        temperature_scaler: temperature scaler
        mu_bias_init: bias initialization for mu
        logvar_init: logvar initialization
        clip_log_var: whether to clip logvar
        clip_vals: values to clip logvar to
        initial_nonlinearities: initial nonlinearity function
        final_nonlinearity: final nonlinearity function
        channel_size: size of the channel
        last_hidden_layer_weight: last hidden layer weight
        last_hidden_layer_bias: last hidden layer bias
        temperature_scaler_mean: temperature scaler mean
        temperature_scaler_log_var: temperature scaler logvar
    """

    def __init__(self, model_params:Dict, dataset_params: Dict) -> None:

        super().__init__()
        self.seq_len = dataset_params["sequence_length"]
        self.alphabet_size = dataset_params["alphabet_size"]
        self.hidden_layer_sizes = model_params["hidden_layer_sizes"]
        self.z_dim = model_params["z_dim"]
        self.dropout_prob = model_params.get("dropout_prob", 0.0)
        self.include_temperature_scaler = model_params.get("include_temperature_scaler", True)
        if self.include_temperature_scaler:
            self.temperature_scaler = nn.Parameter(torch.ones(1))
        self.mu_bias_init = model_params.get("mu_bias_init", 0.1)
        self.logvar_init = model_params.get("logvar_init", -10.0)
        self.clip_log_var = model_params.get("clip_log_var", False)
        if self.clip_log_var:
            self.clip_vals = model_params.get("clip_vals", [-50.0, 50.0])

        self.initial_nonlinearities = nonlinearities_map[model_params.get("initial_non_linearities", "relu")]
        self.final_nonlinearity = nonlinearities_map[model_params.get("final_nonlinearity","relu")]

        hidden_modules: List[nn.Module] = []
        num_features: int = self.z_dim

        for i in range(len(self.hidden_layer_sizes)):
            hidden_modules.append(nn.Linear(num_features, self.hidden_layer_sizes[i]))
            nn.init.constant_(hidden_modules[i].bias, self.mu_bias_init)
            num_features = self.hidden_layer_sizes[i]

        self.hidden_layers = nn.ModuleList(hidden_modules)

    
        if self.dropout_prob > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)

        self.channel_size = self.alphabet_size

        self.last_hidden_layer_weight = nn.Parameter(
            torch.zeros(self.channel_size * self.seq_len, self.hidden_layer_sizes[-1])
        )
        nn.init.xavier_normal_(self.last_hidden_layer_weight)  # Glorot initialization

        self.last_hidden_layer_bias = nn.Parameter(torch.zeros(self.alphabet_size * self.seq_len))
        nn.init.constant_(self.last_hidden_layer_bias, self.mu_bias_init)
    

        if self.include_temperature_scaler:
            self.temperature_scaler_mean = nn.Parameter(torch.ones(1))
            self.temperature_scaler_log_var = nn.Parameter(torch.ones(1) * self.logvar_init)


    def forward(self, z):
        batch_size = z.shape[0]
        if self.dropout_prob > 0.0:
            x = self.dropout_layer(z)
        else:
            x = z

        for i in range(len(self.hidden_layer_sizes) - 1):
            x = self.initial_nonlinearities(self.hidden_layers[i](x))
            if self.dropout_prob > 0.0:
                x = self.dropout_layer(x)

        x = self.final_nonlinearity(self.hidden_layers[-1](x))
        if self.dropout_prob > 0.0:
            x = self.dropout_layer(x)

        W_out = self.last_hidden_layer_weight
        b_out = self.last_hidden_layer_bias

        W_out = W_out.view(self.seq_len * self.alphabet_size, self.hidden_layer_sizes[-1])

        x = F.linear(x, weight=W_out, bias=b_out)

        if self.include_temperature_scaler:
            x = torch.log(1.0 + torch.exp(self.temperature_scaler)) * x

        x = x.view(batch_size, self.seq_len, self.alphabet_size)
        x_recon_log = F.log_softmax(x, dim=-1)  # of shape (batch_size, seq_len, alphabet)

        return x_recon_log


    # the parts of the loss function that need to know the decoder structure or model parameters
    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        return 1

   
    
    def loss_function(
        self,
        x_recon_log,
        x,
        mu,
        log_var,
        kl_latent_scale,
        kl_global_params_scale,
        Neff,
    ):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        """
        # TODO: clip log_var to prevent NaNs
        if self.clip_log_var:
            log_var = torch.clamp(log_var, min=self.clip_vals[0], max=self.clip_vals[1])

        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction="sum") / x.shape[0]
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0]
    
        # warm_up_scale = self.annealing_factor(annealing_warm_up, training_step)
        warm_up_scale = 1.0
        neg_ELBO = BCE + warm_up_scale * (
            kl_latent_scale * KLD_latent
        )

        return neg_ELBO, BCE, KLD_latent, 0.


class BayesianDecoder(nn.Module):
    """
    A bayesian decoder sampling from a distribution  with variance var around a mean value in each 
    layer. The variances are learned as a parameter of the model.
    Args:
        model_params: dictionary of model parameters
        dataset_params: reqiuired information on the dataset 
    Attributes:
        seq_len: length of the sequence
        alphabet_size: size of the alphabet
        hidden_layer_sizes: list of hidden layer sizes
        z_dim: dimension of the latent space
        dropout_prob: dropout probability
        include_temperature_scaler: whether to include a temperature scaler
        temperature_scaler: temperature scaler
        mu_bias_init: bias initialization for mu
        logvar_init: logvar initialization
        clip_log_var: whether to clip logvar
        clip_vals: values to clip logvar to
        initial_nonlinearities: initial nonlinearity function
        final_nonlinearity: final nonlinearity function
        channel_size: size of the channel
        last_hidden_layer_weight: last hidden layer weight
        last_hidden_layer_bias: last hidden layer bias
        temperature_scaler_mean: temperature scaler mean
        temperature_scaler_log_var: temperature scaler logvar
    """

    def __init__(self, model_params, dataset_params: Dict[str, int]) -> None:
        super().__init__()
        self.seq_len = dataset_params["sequence_length"]
        self.alphabet_size = dataset_params["alphabet_size"]
        
        self.hidden_layer_sizes = model_params["hidden_layer_sizes"]
        self.z_dim = model_params["z_dim"]
        self.dropout_prob = model_params.get("dropout_prob", 0.0)
        self.include_temperature_scaler = model_params.get("include_temperature_scaler", True)
        self.mu_bias_init = model_params.get("mu_bias_init", 0.1)
        self.logvar_init = model_params.get("logvar_init", -10.0)
        self.clip_log_var = model_params.get("clip_log_var", False)
        if self.clip_log_var:
            self.clip_vals = model_params.get("clip_vals", [-50.0, 50.0])

        hidden_mean_modules: List[nn.Module] = []
        hidden_log_var_modules: List[nn.Module] = []
        num_features: int = self.z_dim

        for i in range(len(self.hidden_layer_sizes)):
            hidden_mean_modules.append(nn.Linear(num_features, self.hidden_layer_sizes[i]))
            hidden_log_var_modules.append(nn.Linear(num_features, self.hidden_layer_sizes[i]))
            nn.init.constant_(hidden_mean_modules[i].bias, self.mu_bias_init)
            nn.init.constant_(hidden_log_var_modules[i].weight, self.logvar_init)
            num_features = self.hidden_layer_sizes[i]

        self.hidden_layers_mean = nn.ModuleList(hidden_mean_modules)
        self.hidden_layers_log_var = nn.ModuleList(hidden_log_var_modules)

        self.initial_nonlinearities = nonlinearities_map[model_params.get("initial_non_linearities", "relu")]
        self.final_nonlinearity = nonlinearities_map[model_params.get("final_nonlinearity", "relu")]

        if self.dropout_prob > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)

        
        self.last_hidden_layer_weight_mean = nn.Parameter(
            torch.zeros(self.channel_size * self.seq_len, self.hidden_layer_sizes[-1])
        )
        self.last_hidden_layer_weight_log_var = nn.Parameter(
            torch.zeros(self.channel_size * self.seq_len, self.hidden_layer_sizes[-1])
        )
        nn.init.xavier_normal_(self.last_hidden_layer_weight_mean)  # Glorot initialization
        nn.init.constant_(self.last_hidden_layer_weight_log_var, self.logvar_init)

        self.last_hidden_layer_bias_mean = nn.Parameter(torch.zeros(self.alphabet_size * self.seq_len))
        self.last_hidden_layer_bias_log_var = nn.Parameter(torch.zeros(self.alphabet_size * self.seq_len))
        nn.init.constant_(self.last_hidden_layer_bias_mean, self.mu_bias_init)
        nn.init.constant_(self.last_hidden_layer_bias_log_var, self.logvar_init)

        if self.include_temperature_scaler:
            self.temperature_scaler_mean = nn.Parameter(torch.ones(1))
            self.temperature_scaler_log_var = nn.Parameter(torch.ones(1) * self.logvar_init)

    
    def sampler(self, mean, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mean)
        z = torch.exp(0.5 * log_var) * eps + mean
        return z

    def forward(self, z):
        batch_size = z.shape[0]
        if self.dropout_prob > 0.0:
            x = self.dropout_layer(z)
        else:
            x = z

        for i in range(len(self.hidden_layer_sizes) - 1):
            layer_i_weight = self.sampler(
                self.hidden_layers_mean[i].weight,
                self.hidden_layers_log_var[i].weight,
            )
            layer_i_bias = self.sampler(
                self.hidden_layers_mean[i].bias,
                self.hidden_layers_log_var[i].bias,
            )
            x = self.initial_nonlinearities(F.linear(x, weight=layer_i_weight, bias=layer_i_bias))
            if self.dropout_prob > 0.0:
                x = self.dropout_layer(x)

        last_index = len(self.hidden_layers_sizes) - 1
        last_layer_weight = self.sampler(self.hidden_layers_mean[-1].weight, self.hidden_layers_log_var[-1].weight)
        last_layer_bias = self.sampler(self.hidden_layers_mean[-1].bias, self.hidden_layers_log_var[-1].bias)
        x = self.final_nonlinearity(F.linear(x, weight=last_layer_weight, bias=last_layer_bias))
        if self.dropout_prob > 0.0:
            x = self.dropout_layer(x)

        W_out = self.sampler(self.last_hidden_layer_weight_mean, self.last_hidden_layer_weight_log_var)
        b_out = self.sampler(self.last_hidden_layer_bias_mean, self.last_hidden_layer_bias_log_var)

        W_out = W_out.view(self.seq_len * self.alphabet_size, self.hidden_layer_sizes[-1])

        x = F.linear(x, weight=W_out, bias=b_out)

        if self.include_temperature_scaler:
            temperature_scaler = self.sampler(self.temperature_scaler_mean, self.temperature_scaler_log_var)
            x = torch.log(1.0 + torch.exp(temperature_scaler)) * x

        x = x.view(batch_size, self.seq_len, self.alphabet_size)
        x_recon_log = F.log_softmax(x, dim=-1)  # of shape (batch_size, seq_len, alphabet)

        return x_recon_log

    
    # the parts of the loss function that need to know the decoder structure or model parameters
    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        return 1

    
    def KLD_global_parameters(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        KLD_decoder_params = 0.0
        zero_tensor = torch.tensor(0.0)

        for layer_index in range(len(self.hidden_layer_sizes)):
            for param_type in ["weight", "bias"]:
                KLD_decoder_params = KLD_decoder_params + KLD_diag_gaussians(
                    self.state_dict(keep_vars=True)[
                        "hidden_layers_mean." + str(layer_index) + "." + param_type
                    ].flatten(),
                    self.state_dict(keep_vars=True)[
                        "hidden_layers_log_var." + str(layer_index) + "." + param_type
                    ].flatten(),
                    zero_tensor,
                    zero_tensor,
                )

        for param_type in ["weight", "bias"]:
            KLD_decoder_params = KLD_decoder_params + KLD_diag_gaussians(
                self.state_dict(keep_vars=True)["last_hidden_layer_" + param_type + "_mean"].flatten(),
                self.state_dict(keep_vars=True)["last_hidden_layer_" + param_type + "_log_var"].flatten(),
                zero_tensor,
                zero_tensor,
            )

        if self.include_temperature_scaler:
            KLD_decoder_params += KLD_diag_gaussians(
                self.state_dict(keep_vars=True)["temperature_scaler_mean"].flatten(),
                self.state_dict(keep_vars=True)["temperature_scaler_log_var"].flatten(),
                zero_tensor,
                zero_tensor,
            )
        return KLD_decoder_params

    def loss_function(
        self,
        x_recon_log,
        x,
        mu,
        log_var,
        kl_latent_scale,
        kl_global_params_scale,
        Neff,
    ):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        """
    
        if self.clip_log_var:
            log_var = torch.clamp(log_var, min= self.clip_vals[0], max= self.clip_vals[1])

        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction="sum") / x.shape[0]
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0]
        KLD_decoder_params_normalized = self.KLD_global_parameters() / Neff

        #warm_up_scale = self.annealing_factor(annealing_warm_up, training_step)
        warm_up_scale = 1.0
        neg_ELBO = BCE + warm_up_scale * (
            kl_latent_scale * KLD_latent + kl_global_params_scale * KLD_decoder_params_normalized
        )

        return neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized
        

