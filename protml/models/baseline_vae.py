import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Optional

from .util import sample_latent


class VAE(pl.LightningModule):
    """ Variational autoencoder model that learns to generate sequences with a stability similar to wt
        
        Args:
            datamodule: pytorch lightning datamodule with sequences and labels           
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

    def __init__(self, datamodule, encoder, decoder, optimizer, scheduler=None) -> None:
        super().__init__()

        self.dataset = datamodule
        dataset_params = {
            "sequence_length": self.dataset.sequence_length,
            "alphabet_size": self.dataset.alphabet_size
        }

        self.encoder = hydra.utils.instantiate(encoder, dataset_params=dataset_params)
        self.decoder = hydra.utils.instantiate(decoder, dataset_params=dataset_params)
        self.optimizer_params = optimizer
        self.scheduler_params = scheduler

        self.save_hyperparameters(logger=False, ignore=["datamodule"])

    #enabeling load from checkpoint, without checkpointing the data module 
    @classmethod
    def factory(cls, datamodule, **kwargs):
        if "ckpt_path" in kwargs:
            return VAE.load_from_checkpoint(kwargs["ckpt_path"], datamodule=datamodule)
        else:
            return VAE(datamodule=datamodule, **kwargs)

    def forward(self, batch):
        x = batch[0]
        mu, log_var = self.encoder(x)
        z = sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)
        losses = self.decoder.loss_function(recon_x_log, x, mu, log_var, 1.0, 1.0, len(x))
        return losses

    def configure_optimizers(self):
        """Configure the optimizer."""
        self.optimizer = hydra.utils.instantiate(self.optimizer_params, params=self.parameters())
        if self.scheduler_params is None:
            return self.optimizer
        else:
            self.scheduler = hydra.utils.instantiate(self.scheduler_params, optimizer=self.optimizer)
            return self.optimizer, self.scheduler

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        neg_ELBO, _, _, _ = self.forward(batch)
        self.log('train/loss', neg_ELBO, prog_bar=True)
        return {"loss": neg_ELBO}

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        neg_ELBO, _, _, _ = self.forward(batch)
        self.log('val/loss', neg_ELBO, prog_bar=True)
        return {"val/loss": neg_ELBO}

    def on_training_epoch_end(self, outputs: Any) -> None:
        losses = [out["loss"] for out in outputs]

        loss = torch.mean(torch.stack(losses))
        self.log("train/loss", loss)
