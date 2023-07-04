#


## VAE
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L10)
```python 
VAE(
   datamodule, encoder, decoder, optimizer, scheduler = None
)
```


---
Variational autoencoder model that learns to generate sequences with a stability similar to wt


**Args**

* **datamodule**  : pytorch lightning datamodule with sequences and labels           
* **encoder**  : encoder model 
* **measurement**  : measurement model (options: Linear, General Epistasis Model)
* **optimizer**  : optimizer
* **loss_function**  : loss function model( options: )
* **scheduler**  : scheduler


**Attributes**

* **dataset**  : pytorch lightning datamodule
* **encoder**  : encoder model
* **measurement**  : measurement model
* **optimizer**  : optimizer
* **loss_function**  : loss function
* **scheduler**  : scheduler
* **optimizer_params**  : optimizer parameters
* **scheduler_params**  : scheduler parameters



**Methods:**


### .factory
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L50)
```python
.factory(
   cls, datamodule, **kwargs
)
```


### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L56)
```python
.forward(
   batch
)
```


### .configure_optimizers
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L64)
```python
.configure_optimizers()
```

---
Configure the optimizer.

### .training_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L73)
```python
.training_step(
   batch, batch_idx
)
```


### .validation_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L78)
```python
.validation_step(
   batch, batch_idx
)
```


### .on_training_epoch_end
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_vae.py/#L83)
```python
.on_training_epoch_end(
   outputs: Any
)
```

