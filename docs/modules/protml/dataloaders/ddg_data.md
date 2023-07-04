#


## SequenceDataModule
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L21)
```python 
SequenceDataModule(
   datafiles: Dict, params: Dict
)
```


---
A data module to handle sequence data with functional scores


**Args**

* **datafiles** (dict) : A dictionary containing data files. Must contain "train".
* **params** (dict) : Dataloader parameters.


**Attributes**

* **datafile_train** (str) : The path to the training data file.
* **datafile_val** (str) : The path to the validation data file, if provided.
* **datafile_test** (str) : The path to the test data file, if provided.
* **batch_size** (int) : The batch size for the dataloader
* **num_workers** (int) : The number of workers for the dataloader.
* **drop_last** (bool) : Whether to drop the last batch if it is smaller than the batch size.
* **replacement** (bool) : Whether to sample with replacement.
* **sequence_length** (int) : The length of the sequence data.
* **alphabet_size** (int) : The size of the alphabet for the sequence data.



**Methods:**


### .read_data_file
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L63)
```python
.read_data_file(
   filename: str
)
```

---
Reads a csv file containing sequence data and returns a TensorDataset. 
required columns in the csv file are "seq" and "y"

**Args**

* **filename**  : path to csv file


**Returns**

TensorDataset with one-hot encoded sequences and labels

### .setup
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L93)
```python
.setup(
   stage: Optional[str] = None
)
```


### .train_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L121)
```python
.train_dataloader()
```


### .val_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L132)
```python
.val_dataloader()
```


### .test_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L146)
```python
.test_dataloader()
```


----


## Sequence_WT_DataModule
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L159)
```python 
Sequence_WT_DataModule(
   datafiles: Dict, params: Dict
)
```


---
A data module to handle sequence data in the context of the wild type reference sequence.


**Args**

* **datafiles** (dict) : A dictionary containing data files. Must contain "train".
* **params** (dict) : Dataloader parameters.


**Attributes**

* **datafile_train** (str) : The path to the training data file.
* **datafile_val** (str) : The path to the validation data file, if provided.
* **datafile_test** (str) : The path to the test data file, if provided.
* **batch_size** (int) : The batch size for the dataloader
* **num_workers** (int) : The number of workers for the dataloader.
* **drop_last** (bool) : Whether to drop the last batch if it is smaller than the batch size.
* **replacement** (bool) : Whether to sample with replacement.
* **sequence_length** (int) : The length of the sequence data.
* **alphabet_size** (int) : The size of the alphabet for the sequence data.



**Methods:**


### .read_data_file
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L204)
```python
.read_data_file(
   filename: str
)
```

---
Reads a csv file containing sequence data and returns a TensorDataset. 
required columns in the csv file are "seq","wt" and "y"

**Args**

* **filename**  : path to csv file


**Returns**

TensorDataset with one-hot encoded sequences and labels

### .setup
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L230)
```python
.setup(
   stage: Optional[str] = None
)
```


### .train_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L258)
```python
.train_dataloader()
```


### .val_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L269)
```python
.val_dataloader()
```


### .test_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L283)
```python
.test_dataloader()
```


----


## EmbeddingsDataset
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L297)
```python 
EmbeddingsDataset(
   emb_dict, ids: List, *tensors: torch.Tensor
)
```


---
Embeddings Dataset: a dataset overwriting the __getitem__ method to lookup and return the batch embeddings
to avoid loading the entire embeddings file into memory.


**Args**

* **emb_dict**  : dictionary containing the embeddings
* **ids**  : list of protein ids    
* **tensors**  : tensors containing the sequence data and labels


**Attributes**

* **tensors**  : tensors containing the sequence data and labels
* **embeddings**  : dictionary containing the embeddings
* **ids**  : list of protein ids


----


## EmbeddingsDataModule
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L330)
```python 
EmbeddingsDataModule(
   datafiles: Dict, params: Dict
)
```


---
A PyTorch Lightning DataModule for handling sequences embedded with a large protein model (.h5 format).


**Args**

* **datafiles** (dict) : A dictionary containing data files. Must contain "train" and "embeddings".
* **params** (dict) : Dataloader parameters.


**Attributes**

* **embedding_file** (str) : The path to the embedding file.
* **datafile_train** (str) : The path to the training data file.
* **datafile_val** (str) : The path to the validation data file, if provided.
* **datafile_test** (str) : The path to the test data file, if provided.
* **batch_size** (int) : The batch size for the dataloader
* **num_workers** (int) : The number of workers for the dataloader.
* **drop_last** (bool) : Whether to drop the last batch if it is smaller than the batch size.
* **replacement** (bool) : Whether to sample with replacement.
* **sequence_length** (int) : The length of the sequence data.
* **alphabet_size** (int) : The size of the alphabet for the sequence data.



**Methods:**


### .read_data_file
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L375)
```python
.read_data_file(
   filename: str
)
```

---
Reads a csv file containing sequence data and returns a TensorDataset. 
required columns in the csv file are "seq","y" and "id" 

**Args**

* **filename**  : path to csv file


**Returns**

TensorDataset with one-hot encoded sequences and labels

### .generate_embeddings
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L398)
```python
.generate_embeddings(
   df: pd.DataFrame
)
```

---
generate corresponding lookup keys for the input data. We store the keys instead of
all the embeddings due to RAM limitations


**Args**

* **df** (pd.DataFrame) : input data_frame with column "id" containing a unique id-string 
for each sequence that matches the keys in the embedding file 


**Returns**

* **array**  : sequence embeddings wih dim (Nsequences, N_AminoAcids, N_embedding_dim)


### .setup
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L412)
```python
.setup(
   stage: Optional[str] = None
)
```


### .train_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L439)
```python
.train_dataloader()
```


### .val_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L450)
```python
.val_dataloader()
```


### .test_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L464)
```python
.test_dataloader()
```


----


## Embeddings_WT_Dataset
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L477)
```python 
Embeddings_WT_Dataset(
   emb_dict, wt_dict, ids: List, wt: List, *tensors: torch.Tensor
)
```


---
Embeddings Dataset: a dataset that looks up embeddings and the embeddings of the WT sequ in the get_item method


----


## Embeddings_WT_DataModule
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L504)
```python 
Embeddings_WT_DataModule(
   datafiles: dict, params: dict
)
```


---
A PyTorch Lightning DataModule for handling sequences embedded with a large protein model in context
of the corresponding wt sequence.


**Args**

* **datafiles** (dict) : A dictionary containing data files. Must contain "train" and "embeddings".
* **params** (dict) : Dataloader parameters.


**Attributes**

* **embedding_file** (str) : The path to the embedding file.
* **datafile_train** (str) : The path to the training data file.
* **datafile_val** (str) : The path to the validation data file, if provided.
* **datafile_test** (str) : The path to the test data file, if provided.
* **batch_size** (int) : The batch size for the dataloader
* **num_workers** (int) : The number of workers for the dataloader.
* **drop_last** (bool) : Whether to drop the last batch if it is smaller than the batch size.
* **replacement** (bool) : Whether to sample with replacement.
* **sequence_length** (int) : The length of the sequence data.
* **alphabet_size** (int) : The size of the alphabet for the sequence data.



**Methods:**


### .read_data_file
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L553)
```python
.read_data_file(
   filename: str
)
```


### .generate_embeddings
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L570)
```python
.generate_embeddings(
   df: pd.DataFrame
)
```

---
generate corresponding lookup keys for the input data. We store the keys instead of
all the embeddings due to RAM limitations


**Args**

* **df** (pd.DataFrame) : input data_frame with column "id" containing a unique id-string 
for each sequence that matches the keys in the embedding file 


**Returns**

* **array**  : sequence embeddings wih dim (Nsequences, N_AminoAcids, N_embedding_dim)


### .setup
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L592)
```python
.setup(
   stage: Optional[str] = None
)
```


### .train_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L619)
```python
.train_dataloader()
```


### .val_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L630)
```python
.val_dataloader()
```


### .test_dataloader
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/ddg_data.py/#L644)
```python
.test_dataloader()
```

