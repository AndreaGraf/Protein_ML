 Supervised pytorch lightning models with an encoder and a 
    supervised measurement to predict scores

## ENC_M
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L13)
```python 
ENC_M(
   datamodule, encoder, measurement, optimizer, loss_function, scheduler = None
)
```


---
Supervised pytorch lightning model with an encoder and a measurement/prediction model


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
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L56)
```python
.factory(
   cls, datamodule, **kwargs
)
```


### .configure_optimizers
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L62)
```python
.configure_optimizers()
```

---
Configure the optimizer.

### .forward
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L72)
```python
.forward(
   x
)
```


### .step_loss
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L78)
```python
.step_loss(
   batch
)
```


### .training_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L85)
```python
.training_step(
   batch, batch_idx
)
```


### .validation_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L91)
```python
.validation_step(
   batch, batch_idx
)
```


### .test_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L97)
```python
.test_step(
   batch, batch_idx
)
```


### .predict_step
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/models/baseline_supervised.py/#L103)
```python
.predict_step(
   batch, batch_idx
)
```

