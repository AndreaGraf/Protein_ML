import csv
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

import automl.utils as utils

log = utils.get_logger(__name__)


def __parse_range(record_range: List[Union[int, str]], sequence_length: int):
    high = record_range[1]
    if high == ":":
        high = sequence_length

    assert 0 < high <= sequence_length

    low = record_range[0]
    if low == ":":
        low = 0

    assert 0 <= low < high

    return low, high


def output_csv(
    data: pd.DataFrame,
    output_path: str,
    record_range: Optional[List[int]] = None,
    original_csv_input: Optional[str] = None,
):

    output_df = data
    if original_csv_input is not None:
        with open(original_csv_input) as f:
            dialect = csv.Sniffer().sniff(f.read(1024), delimiters=";,|\t")

        original_df = pd.read_csv(original_csv_input, sep=dialect.delimiter, engine="pyarrow")

        if record_range is not None:
            assert len(record_range) == 2, "Expected a sequence of length 2"

        low, high = __parse_range(record_range, len(original_df))
        original_df = original_df.iloc[low:high]
        
        if len(original_df) == len(data):
            original_df.reset_index(drop=True, inplace=True)
            data.reset_index(drop=True, inplace=True)

            output_df = pd.concat([original_df, data], axis=1)
        else:
            log.warning("Original dataframe is not the same length as the output dataframe. Skipping concatenation...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log.info(f"CSV file written to {output_path}")
