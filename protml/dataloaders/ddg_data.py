"""Module for loading and preprocessing ddG data."""

#from omegaconf import OmegaConf
#import omegaconf
from typing import List, Tuple, Dict, Any, Optional
#from collections import defaultdict

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import h5py

from hydra.utils import to_absolute_path
from .utils import (
    generate_ohe_from_sequence_data,
)

class SequenceDataModule(pl.LightningDataModule):
    """
    A data module to handle sequence data with functional scores
    
    Args:
        datafiles (dict): A dictionary containing data files. Must contain "train".
        params (dict): Dataloader parameters.

    Attributes:
        datafile_train (str): The path to the training data file.
        datafile_val (str): The path to the validation data file, if provided.
        datafile_test (str): The path to the test data file, if provided.
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        replacement (bool): Whether to sample with replacement.
        sequence_length (int): The length of the sequence data.
        alphabet_size (int): The size of the alphabet for the sequence data.
    """

    def __init__(self, datafiles:Dict, params:Dict)->None:
        super().__init__()

        self.datafile_train = to_absolute_path(datafiles["train"]) 

        self.datafile_val = datafiles.get("val", None)
        self.datafile_val = to_absolute_path(self.datafile_val) if self.datafile_val is not None else None
        
        self.datafile_test = datafiles.get("test", None)
        self.datafile_test = to_absolute_path(self.datafile_test) if self.datafile_test is not None else None
           

        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)

        self.drop_last = params.get("drop_last", False)
        self.replacement = params.get("replace",True)
        self.sequence_length = params.get("sequence_length")
        self.alphabet_size = params.get("alphabet_size",20)
        self.use_weights = params.get("use_weights", False)


    def read_data_file(self, filename:str)->TensorDataset:
        """
        Reads a csv file containing sequence data and returns a TensorDataset. 
        required columns in the csv file are "seq" and "y"
        Args:
            filename: path to csv file
        Returns: 
            TensorDataset with one-hot encoded sequences and labels
        """

        df = pd.read_csv(filename, index_col=0)
        ohe = generate_ohe_from_sequence_data(np.array(df.seq))
        y = np.array(df.y)

        if self.use_weights:
            self.weights = np.exp(y)
        else:
            self.weights = np.ones_like(y)

        if self.sequence_length is None:
            self.sequence_length = ohe.shape[1]
            self.neff = ohe.shape[0]

        ds = TensorDataset(
            torch.from_numpy(ohe).float(),
            torch.from_numpy(y).float(),
        )
        return ds


    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = self.read_data_file(self.datafile_train)
            if self.datafile_val is not None:
                self.val_data =  self.read_data_file(self.datafile_val)
            else:
                self.val_data = None

        if stage == "test" or stage is None:
            if self.datafile_test is not None:
                self.test_data = self.read_data_file(self.datafile_test) 
            else:
                self.val_data = None


    # define the dataloaders for each stage
    def _dataloader(self, dataset: Dataset, datasampler: Any, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler = datasampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True
        )


    def train_dataloader(self):
        r_sampler = WeightedRandomSampler(
            weights=self.weights.tolist(),
            num_samples=len(self.train_data),
            replacement=self.replacement
        )
        return self._dataloader(
            self.train_data,
            datasampler=r_sampler,
        )

    def val_dataloader(self):
        if self.val_data is None:
            return None
        #mave loader only, no validation in vae
        r_sampler = RandomSampler( 
            self.val_data,
            num_samples=len(self.val_data),
            replacement=True
        )
        return self._dataloader(
            self.val_data,
            datasampler=r_sampler,
        )

    def test_dataloader(self):
# sourcery skip: no-conditionals-in-tests
        if self.test_data is None:
            return None
        my_sampler = SequentialSampler(
            data_source = self.test_data
        )
        return self._dataloader(
            self.test_data,
            datasampler=my_sampler,
        )


class Sequence_WT_DataModule(pl.LightningDataModule):
    """
    A data module to handle sequence data in the context of the wild type reference sequence.
    
    Args:
        datafiles (dict): A dictionary containing data files. Must contain "train".
        params (dict): Dataloader parameters.

    Attributes:
        datafile_train (str): The path to the training data file.
        datafile_val (str): The path to the validation data file, if provided.
        datafile_test (str): The path to the test data file, if provided.
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        replacement (bool): Whether to sample with replacement.
        sequence_length (int): The length of the sequence data.
        alphabet_size (int): The size of the alphabet for the sequence data.
    """
   
    def __init__(self, datafiles: Dict, params: Dict):
        super().__init__()

        self.datafile_train = to_absolute_path(datafiles["train"])
        self.datafile_val = datafiles.get("val")
        if self.datafile_val is not None:
            self.datafile_val = to_absolute_path(self.datafile_val)
        self.datafile_test = datafiles.get("test")
        if self.datafile_test is not None:
            self.datafile_test = to_absolute_path(self.datafile_test)

        #self.use_validation = params.get("use_validation_set", True)

        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)

        self.drop_last = params.get("drop_last", False)
        self.replacement = params.get("replace", True)
        self.sequence_length = None
        self.alphabet_size = params.get("alphabet_size", 20)

        self.train_data = self.read_data_file(self.datafile_train)



    def read_data_file(self, filename:str)->TensorDataset:
        """
        Reads a csv file containing sequence data and returns a TensorDataset. 
        required columns in the csv file are "seq","wt" and "y"
        Args:
            filename: path to csv file
        Returns: 
            TensorDataset with one-hot encoded sequences and labels
        """

        df = pd.read_csv(filename, index_col=0) 
        ohe = generate_ohe_from_sequence_data(np.array(df.seq))
        wt_ohe = generate_ohe_from_sequence_data(np.array(df.wt))
        y = np.array(df.y)

        if self.sequence_length is None: 
            self.sequence_length = ohe.shape[1] 
            self.neff = ohe.shape[0]

        ds = TensorDataset(
            torch.from_numpy(ohe).float(),
            torch.from_numpy(wt_ohe).float(),
            torch.from_numpy(y).float(),
        )
        return ds

    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit" or stage is None:
            #self.train_data = self.read_data_file(self.datafile_train)
            if self.datafile_val is not None:
                self.val_data =  self.read_data_file(self.datafile_val)
            else:
                self.val_data = None
            
        if stage == "test" or stage is None:
            if self.datafile_test is not None:
                self.test_data = self.read_data_file(self.datafile_test)
            else:
                self.test_data = None


    # define the dataloaders for each stage
    def _dataloader(self, dataset: Dataset, datasampler: Any, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler = datasampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True
        )


    def train_dataloader(self):
        r_sampler = RandomSampler(
            self.train_data,
            num_samples=len(self.train_data),
            replacement=self.replacement
        )
        return self._dataloader(
            self.train_data,
            datasampler=r_sampler,
        )
       
    def val_dataloader(self):
        if self.val_data is None:
            return None
        #mave loader only, no validation in vae
        r_sampler = RandomSampler( 
            self.val_data,
            num_samples=len(self.val_data),
            replacement=True
        )
        return self._dataloader(
            self.val_data,
            datasampler=r_sampler,
        )

    def test_dataloader(self):

        if self.test_data is None:
            return None
        my_sampler = SequentialSampler(
            data_source = self.test_data
        )
        return self._dataloader(
            self.test_data,
            datasampler=my_sampler,
        )
    


class EmbeddingsDataset(TensorDataset):
    """ Embeddings Dataset: a dataset overwriting the __getitem__ method to lookup and return the batch embeddings
        to avoid loading the entire embeddings file into memory.

        Args:
            emb_dict: dictionary containing the embeddings
            ids: list of protein ids    
            *tensors: tensors containing the sequence data and labels
            
        Attributes:
            tensors: tensors containing the sequence data and labels
            embeddings: dictionary containing the embeddings
            ids: list of protein ids
        """
        
    def __init__(self, emb_dict, ids:List ,*tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.embeddings = emb_dict
        self.ids = ids

    
    def __getitem__(self, index: int)->Tuple:
        #lookup embedding in .h5 embeddings file 
        pid = self.ids[index]
        embedding = np.array(self.embeddings[pid])
        #remove end of sequence token from embedding 
        embedding = embedding[:-1,:]

        return tuple([embedding] + [tensor[index] for tensor in self.tensors]) 



class EmbeddingsDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling sequences embedded with a large protein model (.h5 format).
    
    Args:
        datafiles (dict): A dictionary containing data files. Must contain "train" and "embeddings".
        params (dict): Dataloader parameters.

    Attributes:
        embedding_file (str): The path to the embedding file.
        datafile_train (str): The path to the training data file.
        datafile_val (str): The path to the validation data file, if provided.
        datafile_test (str): The path to the test data file, if provided.
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        replacement (bool): Whether to sample with replacement.
        sequence_length (int): The length of the sequence data.
        alphabet_size (int): The size of the alphabet for the sequence data.
    """
    def __init__(self, datafiles:Dict, params:Dict):
        super().__init__()
        emb_path = to_absolute_path(datafiles["embeddings"])
        self.embedding_file = h5py.File( emb_path, 'r')

        self.datafile_train = to_absolute_path(datafiles["train"])
        self.datafile_val = datafiles.get("val", None) 
        if self.datafile_val is not None:
            self.datafile_val = to_absolute_path(self.datafile_val)
        self.datafile_test = datafiles.get("test", None)
        if self.datafile_test is not None:
            self.datafile_test = to_absolute_path(self.datafile_test)

        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)
        
        self.drop_last = params.get("drop_last", False)  
        self.replacement = params.get("replace",True)
        self.sequence_length = params.get("sequence_length")
        self.alphabet_size =params.get("alphabet_size",20)

        self.train_data = self.read_data_file(self.datafile_train)
        self.neff = self.train_data.__len__()


    def read_data_file(self, filename:str)->EmbeddingsDataset:
        """
        Reads a csv file containing sequence data and returns a TensorDataset. 
        required columns in the csv file are "seq","y" and "id" 
        Args:
            filename: path to csv file
        Returns: 
            TensorDataset with one-hot encoded sequences and labels
        """

        df = pd.read_csv(filename, index_col=0)
        y = np.array(df.y)

        embeddings = self.generate_embeddings(df)

        ds = EmbeddingsDataset(
            self.embedding_file,
            embeddings,
            torch.from_numpy(y).float(),
        )
        return ds


    def generate_embeddings(self, df: pd.DataFrame) -> List:
        """generate corresponding lookup keys for the input data. We store the keys instead of
        all the embeddings due to RAM limitations

        Args:
            df (pd.DataFrame): input data_frame with column "id" containing a unique id-string 
            for each sequence that matches the keys in the embedding file 

        Returns:
            np.array: sequence embeddings wih dim (Nsequences, N_AminoAcids, N_embedding_dim)
        """
        return [r.id +'_' + str(i) for i, r in df.iterrows()]
        

    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit" or stage is None:
            if self.datafile_val is not None:
                self.val_data =  self.read_data_file(self.datafile_val)
            else:
                self.val_data = None
            
        if stage == "test" or stage is None:
            if self.datafile_test is not None:
                self.test_data = self.read_data_file(self.datafile_test) 
            else:
                self.val_data = None


    # define the dataloaders for each stage
    def _dataloader(self, dataset: Dataset, datasampler: Any, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler = datasampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True
        )


    def train_dataloader(self):
        r_sampler = RandomSampler(
            self.train_data,
            num_samples=len(self.train_data),
            replacement=self.replacement
        )
        return self._dataloader(
            self.train_data,
            datasampler=r_sampler,
        )
       
    def val_dataloader(self):
        if self.val_data is None:
            return None
        #mave loader only, no validation in vae
        r_sampler = RandomSampler( 
            self.val_data,
            num_samples=len(self.val_data),
            replacement=True
        )
        return self._dataloader(
            self.val_data,
            datasampler=r_sampler,
        )

    def test_dataloader(self):
        if self.test_data is None:
            return None
        my_sampler = SequentialSampler(
            data_source = self.test_data
        )
        return self._dataloader(
            self.test_data,
            datasampler=my_sampler,
        )



class Embeddings_WT_Dataset(TensorDataset):
    """ Embeddings Dataset: a dataset that looks up embeddings and the embeddings of the WT sequ in the get_item method
    """
        
    def __init__(self, emb_dict, wt_dict, ids:List, wt:List ,*tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.embeddings = emb_dict
        self.wt = wt_dict
        self.ids = ids
        self.wt_ids = wt
    
    def __getitem__(self, index):
        
        #lookup embedding in .h5 file
        pid = self.ids[index]
        wt_id = self.wt_ids[index]
        embedding = np.array(self.embeddings[pid])
        embedding = embedding[:-1,:]

        wt_embedding = np.array(self.wt[wt_id])
        wt_embedding = wt_embedding[:-1,:]

        return tuple([embedding, wt_embedding] + [tensor[index] for tensor in self.tensors]) 
    


class Embeddings_WT_DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling sequences embedded with a large protein model in context
    of the corresponding wt sequence.
    
    Args:
        datafiles (dict): A dictionary containing data files. Must contain "train" and "embeddings".
        params (dict): Dataloader parameters.

    Attributes:
        embedding_file (str): The path to the embedding file.
        datafile_train (str): The path to the training data file.
        datafile_val (str): The path to the validation data file, if provided.
        datafile_test (str): The path to the test data file, if provided.
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        replacement (bool): Whether to sample with replacement.
        sequence_length (int): The length of the sequence data.
        alphabet_size (int): The size of the alphabet for the sequence data.
    """

    

    def __init__(self, datafiles:dict, params:dict):
        super().__init__()
        emb_path = to_absolute_path(datafiles["embeddings"])
        wt_path = to_absolute_path(datafiles["wt_embeddings"])
        self.embedding_file = h5py.File( emb_path, 'r')
        self.wt = h5py.File(wt_path, 'r')

        self.datafile_train = to_absolute_path(datafiles["train"])
        self.datafile_val = datafiles.get("val", None)
        if self.datafile_val is not None:
            self.datafile_val = to_absolute_path(self.datafile_val)
        self.datafile_test = datafiles.get("test", None)
        if self.datafile_test is not None:
            self.datafile_test = to_absolute_path(self.datafile_test)

        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)

        self.drop_last = params.get("drop_last", False)
        self.replacement = params.get("replace",True)
        self.sequence_length = params.get("sequence_length")
        self.alphabet_size =params.get("alphabet_size",20)

   

    def read_data_file(self, filename:str)->Embeddings_WT_Dataset:
        # read the input data
        df = pd.read_csv(filename, index_col=0) 
        y = np.array(df.y)

        embeddings, wt_embeddings = self.generate_embeddings(df)

        ds = Embeddings_WT_Dataset(
            self.embedding_file,
            self.wt,
            embeddings,
            wt_embeddings,
            torch.from_numpy(y).float(),
        )
        return ds


    def generate_embeddings(self, df:pd.DataFrame)-> Tuple:
        """generate corresponding lookup keys for the input data. We store the keys instead of
        all the embeddings due to RAM limitations

        Args:
            df (pd.DataFrame): input data_frame with column "id" containing a unique id-string 
            for each sequence that matches the keys in the embedding file 

        Returns:
            np.array: sequence embeddings wih dim (Nsequences, N_AminoAcids, N_embedding_dim)
        """
            
        wt_list = []
        emb_list = []
        for i,r in df.iterrows():
            emb_list.append(r.id +'_' + str(i))
            wt_list.append(r.id)

        #strip the last token that was added in emb function for TER
        return emb_list, wt_list


    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = self.read_data_file(self.datafile_train)
            if self.datafile_val is not None:
                self.val_data =  self.read_data_file(self.datafile_val)
            else:
                self.val_data = None

        if stage == "test" or stage is None:
            if self.datafile_test is not None:
                self.test_data = self.read_data_file(self.datafile_test)
            else:
                self.val_data = None


    # define the dataloaders for each stage
    def _dataloader(self, dataset: Dataset, datasampler: Any, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler = datasampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True
        )

    def train_dataloader(self):
        r_sampler = RandomSampler(
            self.train_data,
            num_samples=len(self.train_data),
            replacement=self.replacement
        )
        return self._dataloader(
            self.train_data,
            datasampler=r_sampler,
        )

    def val_dataloader(self):
        if self.val_data is None:
            return None
        #mave loader only, no validation in vae
        r_sampler = RandomSampler( 
            self.val_data,
            num_samples=len(self.val_data),
            replacement=True
        )
        return self._dataloader(
            self.val_data,
            datasampler=r_sampler,
        )

    def test_dataloader(self):
        if self.test_data is None:
            return None
        my_sampler = SequentialSampler(
            data_source = self.test_data
        )
        return self._dataloader(
            self.test_data,
            datasampler=my_sampler,
        )
