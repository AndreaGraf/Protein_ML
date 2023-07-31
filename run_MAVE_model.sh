#!/bin/bash

#the models training 

#train single protein models

# random split
# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/1Y0M.pdb_train_rand.csv  val_data=Data/data_sets/single/1Y0M.pdb_val_rand.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/2KGT.pdb_train_rand.csv  val_data=Data/data_sets/single/2KGT.pdb_val_rand.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/2LGW.pdb_train_rand.csv  val_data=Data/data_sets/single/2LGW.pdb_val_rand.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/3DKM.pdb_train_rand.csv  val_data=Data/data_sets/single/3DKM.pdb_val_rand.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/5VNT.pdb_train_rand.csv  val_data=Data/data_sets/single/5VNT.pdb_val_rand.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# # by residue split
# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/1Y0M.pdb_train_res.csv val_data=Data/data_sets/single/1Y0M.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/2KGT.pdb_train_res.csv val_data=Data/data_sets/single/2KGT.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/2LGW.pdb_train_res.csv val_data=Data/data_sets/single/2LGW.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/3DKM.pdb_train_res.csv val_data=Data/data_sets/single/3DKM.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/5VNT.pdb_train_res.csv val_data=Data/data_sets/single/5VNT.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last

        
# Try different loss functions and measurements (use per-residue split, this makes most sense)
# python3 -m protml.apps.train experiment=supervised/train_base \
#     train_data=Data/data_sets/single/1Y0M.pdb_train_res.csv val_data=Data/data_sets/single/1Y0M.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
#         callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last\
#         model/measurement=linear_measurement 

python3 -m protml.apps.train experiment=supervised/train_base \
    train_data=Data/data_sets/single/5VNT.pdb_train_res.csv val_data=Data/data_sets/single/5VNT.pdb_val_res.csv \
        trainer.max_epochs=5000 model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100] z_dim=1 seed=43\
        callbacks.early_stopping.patience=1000 callbacks.model_checkpoint.save_last=last\
        model.measurement=linear_measurement model/loss_function=gaussnll



# #train a single molecule VAE model 
# python3 -m protml.apps.train experiment=vae/train_base \
#     train_data=Data/data_sets/1Y0M.pdb_train_res.csv val_data=Data/data_sets/1Y0M.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.optimizer.lr=1e-4 model.encoder.model_params.hidden_layer_sizes=[1000,500,100]\
#         model.decoder.model_params.hidden_layer_sizes=[100,500,1000] z_dim=50 seed=42 +datamodule.params.use_weights=True\
#         callbacks.early_stopping.patience=1000  callbacks.early_stopping.patience=1000
        
# python3 -m protml.apps.train experiment=vae/train_base \
#     train_data=Data/data_sets/5VNT.pdb_train_res.csv  val_data=Data/data_sets/5VNT.pdb_val_res.csv \
#         trainer.max_epochs=5000 model.optimizer.lr=1e-4 model.encoder.model_params.hidden_layer_sizes=[1000,500,100]\
#         model.decoder.model_params.hidden_layer_sizes=[100,500,1000] z_dim=10 seed=42 +datamodule.params.use_weights=True\
#         callbacks.early_stopping.patience=1000  callbacks.early_stopping.patience=1000
        
