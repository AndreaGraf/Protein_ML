# protml module documentation

protml uses [Hydra](https://hydra.cc/) for experiment configuration management to allow for quick and convenient experiment configuration and hyperparameter optimization. Hydra allows configuration of basic model parameters in hierachically structured .yaml files, that can be combined as well as overrideen from the commandline to easily modify individual hyperparameters.  

For example, the configurations for the ML models would contain YAML descriptions for all individual components. For example, a simmple MLP `encoder` component looks as follows:

```yaml
_target_: protml.models.encoders.VariationalEncoder
model_params:
  hidden_layer_sizes: [100, 100, 50]
  z_dim: ${z_dim}
  nonlinear_activation: "relu"
  dropout_prob: 0.0
  mu_bias_init: 0.1
  log_var_bias_init: -10.0
```


```console
python3 -m protml.apps.train experiment=supervised/train_base \
    train_data= < PATH_TO_TRAINING_DATA >\
     val_data=D< PATH_TO_VALIDATION_DATA > \
        trainer.max_epochs=50000\
        model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=10
```


