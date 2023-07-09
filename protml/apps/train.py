"""The *protml app* for training models to map protein sequences to their phenotype and
generative models for generate sequences with high functional scores  

Examples:
    Train a supervised model:
        ```python
        python3 -m protml.apps.train experiment=supervised/train_base \
                train_data= < PATH_TO_TRAINING_DATA >\
                val_data= < PATH_TO_VALIDATION_DATA > 
        ```
    Overide model parameters from the command line:

        python3 -m protml.apps.train experiment=supervised/train_base \
                train_data= < PATH_TO_TRAINING_DATA >\
                val_data= < PATH_TO_VALIDATION_DATA >\
                trainer.max_epochs=50000\
                model.encoder.model_params.hidden_layer_sizes=[100,100,100,100,100]\ 
                z_dim=10    

    Train a generative model:
    
        python3 -m protml.apps.train experiment=vae/train_base \
                train_data=< PATH_TO_TRAINING_DATA >\
                val_data= <PATH_TO_VALIDATION_DATA >\
                trainer.max_epochs=1000
                
    Specify parameters that are not set in the config file with +PARAMETER=VALUE:

        python3 -m protml.apps.train experiment=vae/train_base \
                train_data=< PATH_TO_TRAINING_DATA >\
                val_data= <PATH_TO_VALIDATION_DATA >\
                trainer.max_epochs=1000\
                +datamodule.params.use_weights=True
                       
"""

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

import os
from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything

import protml.utils as utils

log = utils.get_logger(__name__)


def log_hyperparameters(config: DictConfig, logger: Any) -> None:
    hparams = dict(config)
    hparams["current_working_dir"] = os.getcwd()
    logger.log_hyperparams(hparams)


def load_data(config: DictConfig) -> LightningDataModule:
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)
    return datamodule


def train(config: DictConfig, datamodule: LightningDataModule) -> None:
    if config.get("seed"):
        log.info("Seeding with {}".format(config.seed))
        seed_everything(config.seed, workers=True)

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model, datamodule=datamodule, _recursive_=False)

    logger = None
    if "logger" in config:
        log.info(f"Instantiating logger <{config.logger._target_}>")
        logger = hydra.utils.instantiate(config.logger, _recursive_=False)
        # Log the interesting training parameters.
        log_hyperparameters(config, logger)

    # Initialize the callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")
    training_flag: bool = config.get("train", True)
    if training_flag:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))

    hparams_search = config.get("hparams_search")
    if hparams_search is not None:
        optimized_metric = hparams_search.get("optimized_metric")
        score = trainer.callback_metrics.get(optimized_metric)
        if score:
            return score.item()
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def do_train(config):

    datamodule = load_data(config)
    _ = train(config, datamodule)


if __name__ == "__main__":
    do_train()
