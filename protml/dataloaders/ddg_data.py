from typing import Dict, Any, Optional
from omegaconf import OmegaConf
import omegaconf
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, SequentialSampler 
import pytorch_lightning as pl
from .utils import alphabet_map
import numpy as np
import pandas as pd
import h5py
from typing import Dict, OrderedDict as OrderedDictType, List, Tuple
from collections import defaultdict
from hydra.utils import to_absolute_path
from .utils import (
    generate_ohe_from_sequence_data,
)

class SequenceDataModule(pl.LightningDataModule):
    """A general data module to handle batches of sequnence data. 
        Args: datafiles: dictionary containing datafiles. Must contain "train"
        params: dataloader parameters
    """
    
    def __init__(self, datafiles:dict, params:dict):
        super().__init__()

        self.datafile_train = to_absolute_path(datafiles.get("train"))
        self.datafile_val = datafiles.get("val", None) 
        if self.datafile_val is not None:
            self.datafile_val = to_absolute_path(self.datafile_val)
        self.datafile_test = datafiles.get("test", None)  
        if self.datafile_test is not None:
            self.datafile_test = to_absolute_path(self.datafile_test)

        #self.use_validation = params.get("use_validation_set", True)
        
        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)
        
        self.drop_last = params.get("drop_last", False)   
        self.replacement = params.get("replace",True)
        self.sequence_length = params.get("sequence_length")
        self.alphabet_size = params.get("alphabet_size",20)
        self.use_weights = params.get("use_weights", False)

        #self.train_data = self.read_data_file(self.datafile_train)


    def read_data_file(self, filename:str)->TensorDataset:
        df = pd.read_csv(filename, index_col=0) 
        ohe = generate_ohe_from_sequence_data(np.array(df.seq))
        y = np.array(df.y)
        
        if self.use_weights:
            self.weights = np.exp(y)
        else:
            self.weights = np.ones_like(y) 

        if self.sequence_length is None: 
            self.sequence_length = ohe.shape[1] 
            self.Neff = ohe.shape[0]

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
            weights=self.weights,
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


class Sequence_WT_DataModule(pl.LightningDataModule):
    """ Sequence data module to handle batches of sequnence data with a copy of the wild type reference sequence. 
        Args: datafiles: dictionary containing datafiles. Must contain "train"
        params: dataloader parameters
    """
    def __init__(self, datafiles:dict, params:dict):
        super().__init__()

        self.datafile_train = to_absolute_path(datafiles.get("train"))
        self.datafile_val = datafiles.get("val", None) 
        if self.datafile_val is not None:
            self.datafile_val = to_absolute_path(self.datafile_val)
        self.datafile_test = datafiles.get("test", None)   
        if self.datafile_test is not None:
            self.datafile_test = to_absolute_path(self.datafile_test)

        #self.use_validation = params.get("use_validation_set", True)
        
        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)
        
        self.drop_last = params.get("drop_last", False)   
        self.replacement = params.get("replace",True)
        self.sequence_length = None
        self.alphabet_size =params.get("alphabet_size",20)

        self.train_data = self.read_data_file(self.datafile_train)



    def read_data_file(self, filename:str)->TensorDataset:
        df = pd.read_csv(filename, index_col=0) 
        ohe = generate_ohe_from_sequence_data(np.array(df.seq))
        wt_ohe = generate_ohe_from_sequence_data(np.array(df.wt))
        y = np.array(df.y)

        if self.sequence_length is None: 
            self.sequence_length = ohe.shape[1] 
            self.Neff = ohe.shape[0]

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
                self.test_data = self.read_data_file() 
            else:
                self.val_data = None


    
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
    """ Embeddings Dataset: a dataset that looks up embeddings from .h5 in the get_item method
    """
        
    def __init__(self, emb_dict, ids:List ,*tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.embeddings = emb_dict
        self.ids = ids

    
    def __getitem__(self, index):
        
        #lookup embedding in .h5 embeddings file 
        pid = self.ids[index]
        embedding = np.array(self.embeddings[pid])
        embedding = embedding[:-1,:]

        return tuple([embedding] + [tensor[index] for tensor in self.tensors]) 



class EmbeddingsDataModule(pl.LightningDataModule):
    """ Data module to handle batches of embedded sequences from pretrained language models
        Args: datafiles: dictionary containing datafiles. Must contain "train"
        params: dataloader parameters
    """
    
    def __init__(self, datafiles:dict, params:dict):
        super().__init__()
        emb_path = to_absolute_path(datafiles.get("embeddings"))
        self.embedding_file = h5py.File( emb_path, 'r')
  
        self.datafile_train = to_absolute_path(datafiles.get("train"))
        self.datafile_val = datafiles.get("val", None) 
        if self.datafile_val is not None:
            self.datafile_val = to_absolute_path(self.datafile_val)
        self.datafile_test = datafiles.get("test", None)   
        if self.datafile_test is not None:
            self.datafile_test = to_absolute_path(self.datafile_test)

        #self.use_validation = params.get("use_validation_set", True)
        
        self.batch_size = params.get("batch_size", 256)
        self.num_workers = params.get("num_workers", 4)
        
        self.drop_last = params.get("drop_last", False)   
        self.replacement = params.get("replace",True)
        self.sequence_length = params.get("sequence_length")
        self.alphabet_size =params.get("alphabet_size",20)

        self.train_data = self.read_data_file(self.datafile_train)
        self.Neff = self.train_data.__len__()
        

    def read_data_file(self, filename:str)->EmbeddingsDataset:
        df = pd.read_csv(filename, index_col=0) 
        #ohe = generate_ohe_from_sequence_data(np.array(df.seq))
        y = np.array(df.y)

        #sequence length is needed for model_dimension construction
        #if self.sequence_length is None: 
        #    self.Neff = y.shape[0]

        embeddings = self.generate_embeddings(df)

        ds = EmbeddingsDataset(
            self.embedding_file,
            embeddings,
            #torch.from_numpy(ohe).float(),
            torch.from_numpy(y).float(),
        )
        return ds
    
   
    def generate_embeddings(self, df: pd.DataFrame)-> List:
        """generate corresponding lookup keys for the input data. We store the keys instead of
        all the embeddings due to RAM limitations

        Args:
            df (pd.DataFrame): input data_frame

        Returns:
            np.array: sequence embeddings wih dim (Nsequences, N_AminoAcids, N_embedding_dim)
        """

        emb_list = []
        for i,r in df.iterrows():
            emb_list.append(r.pdbid +'_' + str(i))
        
        #strip the last token that was added in emb function for TER
        return emb_list


    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit" or stage is None:
            #self.train_data = self.read_data_file(self.datafile_train)
            if self.datafile_val is not None:
                self.val_data =  self.read_data_file(self.datafile_val)
            else:
                self.val_data = None
            
        if stage == "test" or stage is None:
            if self.datafile_test is not None:
                self.test_data = self.read_data_file() 
            else:
                self.val_data = None


    
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
    """ Data module to handle batches of embedded sequences + corresponding embedded wt sequence 
        from pretrained language models
        Args: datafiles: dictionary containing datafiles. Must contain "train"
        params: dataloader parameters
    """
    
    def __init__(self, datafiles:dict, params:dict):
        super().__init__()
        emb_path = to_absolute_path(datafiles.get("embeddings"))
        wt_path = to_absolute_path(datafiles.get("wt_embeddings"))
        self.embedding_file = h5py.File( emb_path, 'r')
        self.wt = h5py.File(wt_path, 'r')
  
        self.datafile_train = to_absolute_path(datafiles.get("train"))
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

        

    def read_data_file(self, filename:str)->EmbeddingsDataset:
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
    
   
    def generate_embeddings(self, df: pd.DataFrame)-> List:
        """generate corresponding lookup keys for the input data. We store the keys instead of
        all the embeddings due to RAM limitations

        Args:
            df (pd.DataFrame): input data_frame

        Returns:
            np.array: sequence embeddings wih dim (Nsequences, N_AminoAcids, N_embedding_dim)
        """
        wt_list = []
        emb_list = []
        for i,r in df.iterrows():
            emb_list.append(r.pdbid +'_' + str(i))
            wt_list.append(r.pdbid)
        
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
                self.test_data = self.read_data_file() 
            else:
                self.val_data = None


    
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

