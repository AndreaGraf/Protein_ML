Module containing loss functions for training,




## MSE_Loss
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L12)
```python 

```


---
wrapper for a simple MSE loss with the same return shape. Forward function takes

**Args**

yhat, ytrue

**Returns**

MSE loss mean


**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L24)
```python
.forward(
   yhat, ytrue
)
```


----


## GaussNLLLoss
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L28)
```python 

```


---
Wrapper for a Gaussian negative log likelihood loss. Forward: 

**Args**

yhat, ytrue

**Returns**

NLL loss mean


**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L41)
```python
.forward(
   yhat, ytrue
)
```


----


## GaussNLL_VAR0_Loss
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L47)
```python 

```


---
Gaussian negative log likelihood loss with a trainable variance parameter. Forward:

**Args**

yhat, ytrue 

**Returns**

NLL loss mean


**Methods:**


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L63)
```python
.forward(
   yhat, ytrue
)
```


----


## GaussNLL_VAR_Loss
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L68)
```python 

```


---
Gaussian negative log likelihood loss treating logsigma as a second order polynomial expansion 
similar to the noisemodel in MAVE NN. Forward:

**Args**

yhat, ytrue 

**Returns**

NLL loss mean


**Methods:**


### .calc_logsigma
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L86)
```python
.calc_logsigma(
   yhat
)
```

---
function to compute the variance based on a order polinomial expansion

### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L95)
```python
.forward(
   yhat, ytrue
)
```


----


## NoiseLayer
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L104)
```python 
NoiseLayer(
   model_params
)
```


---
Base class for original MAVE-NN noise layers

**Args**

* **model_params** (dict) : Dictionary of model parameters.
* **key**  : polynomial_order


**Attributes**

* **poly_order** (int) : Order of polynomial expansion for noise model.



**Methods:**


### .compute_nlls
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L118)
```python
.compute_nlls(
   yhat, ytrue
)
```

---
Compute the negative log-likelihoods for the given predictions and targets
defined in the derived noise model classes.


**Args**

* **yhat** (torch.Tensor) : Predictions from the model.
* **ytrue** (torch.Tensor) : Targets for the model.


**Returns**

* **Tensor**  : The negative log-likelihoods for each sample in the batch.


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L132)
```python
.forward(
   yhat, ytrue
)
```


----


## GaussianNoise
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L137)
```python 
GaussianNoise(
   model_params
)
```


---
A Gaussian noise distribution for GE regression 

**Args**

* **model_params** (dict) : Dictionary of model parameters.


**Attributes**

* **poly_order** (int) : Order of polynomial expansion for noise model.

---
 Methods:
     compute_nlls: Compute the negative log likelihood using the computed logsigma
 


**Methods:**


### .compute_params
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L156)
```python
.compute_params(
   yhat, y_true = None
)
```

---
Compute layer parameters governing p(y|yhat).

### .compute_nlls
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L167)
```python
.compute_nlls(
   yhat, ytrue
)
```

---
Compute negative log likelihood contributions for each datum.

----


### KLD_diag_gaussians
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/components/loss_functions.py/#L178)
```python
.KLD_diag_gaussians(
   mu: torch.Tensor, logvar: torch.Tensor, p_mu: torch.Tensor,
   p_logvar: torch.Tensor
)
```

---
KL divergence between diagonal gaussian with prior diagonal gaussian.

**Args**

* **mu** (torch.Tensor) : mean of the posterior
* **logvar** (torch.Tensor) : log variance of the posterior
* **p_mu** (torch.Tensor) : mean of the prior
* **p_logvar** (torch.Tensor) : log variance of the prior


**Returns**

KL divergence (torch.Tensor)
