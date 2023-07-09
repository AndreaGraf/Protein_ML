The *protml app* for training models to map protein sequences to their phenotype and
generative models for generate sequences with high functional scores

### Examples
Train a supervised model:

```python
python3 -m protml.apps.train experiment=supervised/train_base train_data= < PATH_TO_TRAINING_DATA > val_data= < PATH_TO_VALIDATION_DATA >
```

Overide model parameters from the command line:

```python
python3 -m protml.apps.train experiment=supervised/train_base train_data= < PATH_TO_TRAINING_DATA > val_data= < PATH_TO_VALIDATION_DATA > trainer.max_epochs=50000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=10
```

Train a generative model:

```python
python3 -m protml.apps.train experiment=vae/train_base train_data=< PATH_TO_TRAINING_DATA > val_data= <PATH_TO_VALIDATION_DATA > trainer.max_epochs=1000
```

Specify parameters that are not set in the config file with +PARAMETER=VALUE:

```python
python3 -m protml.apps.train experiment=vae/train_base train_data=< PATH_TO_TRAINING_DATA > val_data= <PATH_TO_VALIDATION_DATA > trainer.max_epochs=1000 +datamodule.params.use_weights=True
```





### log_hyperparameters
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/apps/train.py/#L53)
```python
.log_hyperparameters(
   config: DictConfig, logger: Any
)
```


----


### load_data
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/apps/train.py/#L59)
```python
.load_data(
   config: DictConfig
)
```


----


### train
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/apps/train.py/#L65)
```python
.train(
   config: DictConfig, datamodule: LightningDataModule
)
```


----


### do_train
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/apps/train.py/#L109)
```python
.do_train(
   config
)
```

