#


## ENC_M_dz
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L10)
```python 
ENC_M_dz(
   datamodule, encoder, measurement, optimizer, loss_function, scheduler = None
)
```


---
Supervised pytorch lightning model with an encoder and a measurement/prediction model
The supervised measurement trains on the Ndim distance between the 
embeddings of a sequence and the reference wildtypd.


**Args**

* **datamodule**  : pytorch lightning datamodule with sequences and labels and wt sequences          
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
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L54)
```python
.factory(
   cls, datamodule, **kwargs
)
```


### .configure_optimizers
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L60)
```python
.configure_optimizers()
```

---
Configure the optimizer.

### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L69)
```python
.forward(
   batch
)
```


### .training_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L79)
```python
.training_step(
   batch, batch_idx
)
```


### .validation_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L85)
```python
.validation_step(
   batch, batch_idx
)
```


### .test_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/dz_supervised.py/#L91)
```python
.test_step(
   batch, batch_idx
)
```

