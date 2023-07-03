from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..util import nonlinearities_map


class VariationalEncoder(nn.Module):
    """ Defining the architecture of a MLP encoder
        Args:
            model_params(dict): dictionary containing the model parameters
            dataset_params(dict): dictionary containing the dataset parameters

        Attributes:
            seq_len(int): length of the input sequence
            alphabet_size(int): size of the alphabet
            hidden_layer_sizes(list): list containing the sizes of the hidden layers
            z_dim(int): dimension of the latent space
            dropout_prob(float): dropout probability
            non_linearity(function): non-linearity function
    """
    def __init__(
        self,
        model_params,
        dataset_params: Dict[str, int],
    ) -> None:
        """
        The encoder maps the input to a latent space.
        """
        super().__init__()
        self.seq_len = dataset_params["sequence_length"]
        self.alphabet_size = dataset_params["alphabet_size"]
    
        self.hidden_layer_sizes = model_params.get("hidden_layer_sizes")
        self.z_dim = model_params.get("z_dim")
        self.dropout_prob = model_params.get("dropout_prob", 0.0)

        self.non_linearity = nonlinearities_map[model_params.get("nonlinear_activation", "relu")]

        self.mu_bias_init = model_params.get("mu_bias_init", 0.1)
        self.log_var_bias_init = model_params.get("log_var_bias_init", -10.0)

        self.channel_size = self.alphabet_size

        hidden_modules = []
        num_features = self.channel_size * self.seq_len
        
        for i in range(len(self.hidden_layer_sizes)):
            hidden_modules.append(nn.Linear(num_features, self.hidden_layer_sizes[i]))
            nn.init.constant_(hidden_modules[i].bias, self.mu_bias_init)
            num_features = self.hidden_layer_sizes[i]

        self.hidden_modules = nn.ModuleList(hidden_modules)

        #initialize the mean and variance of the latent space layer
        self.fc_mean = nn.Linear(self.hidden_layer_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_mean.bias, self.mu_bias_init)
        self.fc_log_var = nn.Linear(self.hidden_layer_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_log_var.bias, self.log_var_bias_init)

        if self.dropout_prob > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        x = x.view(-1, self.seq_len * self.channel_size)

        for layer_index in range(len(self.hidden_layer_sizes)):
            x = self.non_linearity(self.hidden_modules[layer_index](x))
            if self.dropout_prob > 0.0:
                x = self.dropout_layer(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        return z_mean, z_log_var
