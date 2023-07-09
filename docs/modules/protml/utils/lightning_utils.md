Module with utilities for PyTorch Lightning models



### get_logger
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/utils/lightning_utils.py/#L15)
```python
.get_logger(
   name = __name__
)
```

---
Initializes multi-GPU-friendly python command line logger.

**Args**

* **name** (str, optional) : Name of the logger. Defaults to __name__.


**Returns**

Initialized logger.

----


### extras
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/utils/lightning_utils.py/#L44)
```python
.extras(
   config: DictConfig
)
```

---
Applies optional utilities, controlled by config flags.

Utilities:
- Ignoring python warnings
- Rich config printing

----


### print_config
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/utils/lightning_utils.py/#L64)
```python
.print_config(
   config: DictConfig, print_order: Sequence[str] = ('datamodule', 'model',
   'callbacks', 'logger', 'trainer'), resolve: bool = True
)
```

---
Prints content of DictConfig using Rich library and its tree structure.


**Args**

* **config** (DictConfig) : Configuration composed by Hydra.
* **print_order** (Sequence[str], optional) : Determines in what order config 
                                    components are printed.       
* **resolve** (bool, optional) : Whether to resolve
                        reference fields of DictConfig.


----


### log_hyperparameters
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/utils/lightning_utils.py/#L115)
```python
.log_hyperparameters(
   config: DictConfig, model: pl.LightningModule, trainer: pl.Trainer
)
```

---
Controls which config parts are saved by Lightning loggers.


**Args**

* **config** (DictConfig) : Configuration composed by Hydra.   
* **model** (pl.LightningModule) : Lightning model.

