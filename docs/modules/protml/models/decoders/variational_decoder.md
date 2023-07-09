#


## MLPDecoder
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L12)
```python 
MLPDecoder(
   model_params: Dict, dataset_params: Dict
)
```


---
standard multilayer perceptron decoder class

**Args**

* **model_params**  : dictionary of model parameters
* **dataset_params**  : reqiuired information on the dataset 


**Attributes**

* **seq_len**  : length of the sequence
* **alphabet_size**  : size of the alphabet
* **hidden_layer_sizes**  : list of hidden layer sizes
* **z_dim**  : dimension of the latent space
* **dropout_prob**  : dropout probability
* **include_temperature_scaler**  : whether to include a temperature scaler
* **temperature_scaler**  : temperature scaler
* **mu_bias_init**  : bias initialization for mu
* **logvar_init**  : logvar initialization
* **clip_log_var**  : whether to clip logvar
* **clip_vals**  : values to clip logvar to
* **initial_nonlinearities**  : initial nonlinearity function
* **final_nonlinearity**  : final nonlinearity function
* **channel_size**  : size of the channel
* **last_hidden_layer_weight**  : last hidden layer weight
* **last_hidden_layer_bias**  : last hidden layer bias
* **temperature_scaler_mean**  : temperature scaler mean
* **temperature_scaler_log_var**  : temperature scaler logvar



**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L89)
```python
.forward(
   z
)
```


### .annealing_factor
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L122)
```python
.annealing_factor(
   annealing_warm_up, training_step
)
```

---
Annealing schedule of KL to focus on reconstruction error in early stages of training

### .loss_function
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L130)
```python
.loss_function(
   x_recon_log, x, mu, log_var, kl_latent_scale, kl_global_params_scale, Neff
)
```

---
Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.

----


## BayesianDecoder
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L159)
```python 
BayesianDecoder(
   model_params, dataset_params: Dict[str, int]
)
```


---
A bayesian decoder sampling from a distribution  with variance var around a mean value in each 
layer. The variances are learned as a parameter of the model.

**Args**

* **model_params**  : dictionary of model parameters
* **dataset_params**  : reqiuired information on the dataset 


**Attributes**

* **seq_len**  : length of the sequence
* **alphabet_size**  : size of the alphabet
* **hidden_layer_sizes**  : list of hidden layer sizes
* **z_dim**  : dimension of the latent space
* **dropout_prob**  : dropout probability
* **include_temperature_scaler**  : whether to include a temperature scaler
* **temperature_scaler**  : temperature scaler
* **mu_bias_init**  : bias initialization for mu
* **logvar_init**  : logvar initialization
* **clip_log_var**  : whether to clip logvar
* **clip_vals**  : values to clip logvar to
* **initial_nonlinearities**  : initial nonlinearity function
* **final_nonlinearity**  : final nonlinearity function
* **channel_size**  : size of the channel
* **last_hidden_layer_weight**  : last hidden layer weight
* **last_hidden_layer_bias**  : last hidden layer bias
* **temperature_scaler_mean**  : temperature scaler mean
* **temperature_scaler_log_var**  : temperature scaler logvar



**Methods:**


### .sampler
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L242)
```python
.sampler(
   mean, log_var
)
```

---
Samples a latent vector via reparametrization trick

### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L250)
```python
.forward(
   z
)
```


### .annealing_factor
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L295)
```python
.annealing_factor(
   annealing_warm_up, training_step
)
```

---
Annealing schedule of KL to focus on reconstruction error in early stages of training

### .KLD_global_parameters
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L302)
```python
.KLD_global_parameters()
```

---
KL divergence between the variational distributions and the priors (for the decoder weights).

### .loss_function
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/decoders/variational_decoder.py/#L339)
```python
.loss_function(
   x_recon_log, x, mu, log_var, kl_latent_scale, kl_global_params_scale, Neff
)
```

---
Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
