"""A collection of utility functions for handeling protein and DNA sequence data"""

from typing import Dict
import numpy as np


alphabet_map = {"protein": "ACDEFGHIKLMNPQRSTVWY"}


def generate_dict_from_alphabet(alphabet: str) -> Dict[str, int]:
    """Map the alphabet letters to their positions with a dictionary. Serves as a
    helper function for one-hot encoding.
    Args:
        alphabet (str): Input dictionary for the specific class of molecules.

    Returns:
        Dict[str, int]: Mapped dictionary.
    """
    return {letter:i for i, letter in enumerate(alphabet)}


def generate_ohe_from_sequence_data(
    sequences: np.array,
    molecule_to_number: Dict = generate_dict_from_alphabet(alphabet_map["protein"]),
) -> np.ndarray:
    """generate one hot encoded data from a sequence"""

    # one hot encoding:
    seq_ohe= np.zeros((sequences.shape[0], len(sequences[0]),
                        len(molecule_to_number)),dtype = np.float32)
    for i, seq in enumerate(sequences):
        for j, letter in enumerate(seq):
            if letter in molecule_to_number:
                k = molecule_to_number[letter]
                seq_ohe[i, j, k] = 1.0

    return seq_ohe

def pad_sequence(seq: str, pad_to: int)->str:
    """ Pad (or truncate) sequence to specified length
    Args:
        seq (str): AA sequence
        pad_to (int): target length
    Returns:
        str: padded sequence
    """

    return seq.ljust(pad_to, '-' ) if len(seq) <= pad_to else seq[:pad_to]
