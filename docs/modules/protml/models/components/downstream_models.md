#


## Mave_Global_Epistasis_Measurement
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/downstream_models.py/#L8)
```python 
Mave_Global_Epistasis_Measurement(
   model_params
)
```


---
Nonlinear GE Measurement process. for z_dim = 1 this is the same as in MAVE-NN

**Args**

* **model_params** (dict) : Dictionary of model parameters
* **keys**  : num_hidden_nodes, z_dim                    


**Attributes**

* **num_hidden_nodes** (int) : Number of hidden nodes in the measurement process.
* **z_dim** (int) : Dimension of the latent space.



**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/downstream_models.py/#L32)
```python
.forward(
   z
)
```

---
Compute y_hat from sample of z

----


## Linear_Measurement
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/downstream_models.py/#L39)
```python 
Linear_Measurement(
   model_params
)
```


---
Linear Measurement head mapping z to yhat

**Args**

* **model_params** (dict) : Dictionary of model parameters
* **keys**  : z_dim, mu_bias_init(optional)                    


**Attributes**

* **num_hidden_nodes** (int) : Number of hidden nodes in the measurement process.
* **z_dim** (int) : Dimension of the latent space.



**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/downstream_models.py/#L59)
```python
.forward(
   yh
)
```

