from typing import Any, Optional
import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .util import sample_latent


class ENC_M_dz(pl.LightningModule):
    """ Supervised pytorch lightning model with an encoder and a measurement/prediction model
            The supervised measurement trains on the Ndim distance between the 
            embeddings of a sequence and the reference wildtypd.
        
        Args:
            datamodule: pytorch lightning datamodule with sequences and labels and wt sequences          
            encoder: encoder model 
            measurement: measurement model (options: Linear, General Epistasis Model)
            optimizer: optimizer
            loss_function: loss function model( options: )
            scheduler: scheduler

        Attributes:
            dataset: pytorch lightning datamodule
            encoder: encoder model
            measurement: measurement model
            optimizer: optimizer
            loss_function: loss function
            scheduler: scheduler
            optimizer_params: optimizer parameters
            scheduler_params: scheduler parameters  
    """

    def __init__(self, datamodule, encoder, measurement, optimizer, loss_function, scheduler=None) -> None:
        super().__init__()

        self.dataset = datamodule
        dataset_params = {
            "sequence_length": self.dataset.sequence_length,
            "alphabet_size": self.dataset.alphabet_size,
        }

        self.encoder = hydra.utils.instantiate(encoder, dataset_params=dataset_params)
        self.measurement = hydra.utils.instantiate(measurement)
        self.loss_function = hydra.utils.instantiate(loss_function)

        self.optimizer_params = optimizer
        self.scheduler_params = scheduler

        self.save_hyperparameters(logger=False, ignore=["datamodule"])

    #enabeling load from checkpoint, without data module
    @classmethod
    def factory(cls, datamodule, **kwargs):
        if "ckpt_path" in kwargs:
            return ENC_M_dz.load_from_checkpoint(kwargs["ckpt_path"], datamodule=datamodule)
        else:
            return ENC_M_dz(datamodule=datamodule, **kwargs)
        
    def configure_optimizers(self):
        """Configure the optimizer."""
        self.optimizer = hydra.utils.instantiate(self.optimizer_params, params=self.parameters())
        if self.scheduler_params is None:
            return self.optimizer
        else:
            self.scheduler = hydra.utils.instantiate(self.scheduler_params, optimizer=self.optimizer)
            return self.optimizer, self.scheduler

    def forward(self, batch):
        x,wt,y = batch
        mu, _ = self.encoder(x)
        mu_wt,_ = self.encoder(wt)
        dz = mu - mu_wt
        y_hat = self.measurement(dz)
        loss = self.loss_function(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        mp_loss = self.forward(batch)
        self.log('train/loss', mp_loss, prog_bar=True)
        return {"loss": mp_loss}


    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        mp_loss = self.forward(batch)
        self.log('val/loss', mp_loss, prog_bar=True)
        return {"loss": mp_loss}
 

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        mp_loss = self.forward(batch)
        self.log('test/loss', mp_loss)
        return {"loss": mp_loss}
    
 