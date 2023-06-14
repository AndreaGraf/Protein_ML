#!/bin/bash


#train a single protein model

#python3 -m automl.apps.train experiment=supervised/train_base \
#    train_data=Data/data_sets/df_train_rand_1MJC.csv val_data=Data/data_sets/df_val_rand_1MJC.csv \
#        trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#        callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last
        
python3 -m automl.apps.train experiment=supervised/train_base \
    train_data=Data/data_sets/df_train_rand_1MJC.csv val_data=Data/data_sets/df_val_rand_1MJC.csv \
        trainer.max_epochs=50000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=10 seed=43\
        callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last 

python3 -m automl.apps.train experiment=supervised/train_base model=lin_mave\
    train_data=Data/data_sets/df_train_rand_1MJC.csv val_data=Data/data_sets/df_val_rand_1MJC.csv \
        trainer.max_epochs=50000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
        callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last \
        
        

#train a single molecule VAE model 

python3 -m automl.apps.train experiment=vae/train_base \
    train_data=Data/data_sets/df_train_rand_1MJC.csv val_data=Data/data_sets/df_val_rand_1MJC.csv \
        trainer.max_epochs=5000 model.optimizer.lr=1e-4 model.encoder.model_params.hidden_layer_sizes=[1000,500,100]\
        model.decoder.model_params.hidden_layer_sizes=[100,500,1000] z_dim=50 seed=42 +datamodule.params.use_weights=True\
        callbacks.early_stopping.patience=1000  callbacks.early_stopping.patience=1000
        
python3 -m automl.apps.train experiment=vae/train_base \
    train_data=Data/data_sets/df_train_rand_1MJC.csv val_data=Data/data_sets/df_val_rand_1MJC.csv \
        trainer.max_epochs=5000 model.optimizer.lr=1e-4 model.encoder.model_params.hidden_layer_sizes=[1000,500,100]\
        model.decoder.model_params.hidden_layer_sizes=[100,500,1000] z_dim=10 seed=42 +datamodule.params.use_weights=True\
        callbacks.early_stopping.patience=1000  callbacks.early_stopping.patience=1000
        
