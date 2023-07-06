#


## VariationalEncoder
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/encoders/variational_encoder.py/#L9)
```python 
VariationalEncoder(
   model_params, dataset_params: Dict[str, int]
)
```


---
Defining the architecture of a MLP encoder

**Args**

* **model_params** (dict) : dictionary containing the model parameters
* **dataset_params** (dict) : dictionary containing the dataset parameters


**Attributes**

* **seq_len** (int) : length of the input sequence
* **alphabet_size** (int) : size of the alphabet
* **hidden_layer_sizes** (list) : list containing the sizes of the hidden layers
* **z_dim** (int) : dimension of the latent space
* **dropout_prob** (float) : dropout probability
* **non_linearity** (function) : non-linearity function



**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/encoders/variational_encoder.py/#L65)
```python
.forward(
   x: torch.Tensor
)
```

