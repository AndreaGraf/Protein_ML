Utility functions used by different models

### get_l2_norm
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/util.py/#L18)
```python
.get_l2_norm(
   model: torch.nn.Module
)
```

---
Compute the L2 norm of the module weights.

**Args**

* **model**  : pytorch module - the module to calculate the l2 norm for


----


### sample_latent
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/util.py/#L28)
```python
.sample_latent(
   mu: torch.Tensor, log_var: torch.Tensor
)
```

---
Samples a latent vector via reparametrization trick

**Args**

* **mu** (torch.Tensor) : mean of the latent distribution
* **log_var** (torch.Tensor) : log variance of the latent distribution


**Returns**

* **z** (torch.Tensor) : latent vector

