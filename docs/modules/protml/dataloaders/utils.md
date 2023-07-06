#


### generate_dict_from_alphabet
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/utils.py/#L10)
```python
.generate_dict_from_alphabet(
   alphabet: str
)
```

---
Map the alphabet letters to their positions with a dictionary. Serves as a
helper function for one-hot encoding.

**Args**

* **alphabet** (str) : Input dictionary for the specific class of molecules.


**Returns**

* Mapped dictionary.


----


### generate_ohe_from_sequence_data
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/utils.py/#L22)
```python
.generate_ohe_from_sequence_data(
   sequences: np.array, molecule_to_number: Dict = None
)
```

---
generate one hot encoded data from a sequence

**Args**

* **sequences** (np.array) : Input array of sequences.
* **molecule_to_number** (Dict) : Dictionary mapping the alphabet to their positions.


**Returns**

* **ndarray**  : One hot encoded array.


----


### pad_sequence
[source](https://github.com/AndreaGraf/Protein_ML/blob/read_the_docs/protml/dataloaders/utils.py/#L48)
```python
.pad_sequence(
   seq: str, pad_to: int
)
```

---
Pad (or truncate) sequence to specified length

**Args**

* **seq** (str) : AA sequence
* **pad_to** (int) : target length


**Returns**

* **str**  : padded sequence

