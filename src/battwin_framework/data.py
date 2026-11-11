"""Load bundled or user-supplied data without changing values or aggregations."""

from pathlib import Path
import pandas as pd
from .resources import ASSETS

DATASETS = {
    "test": "test_aggregated_deterministic.xlsx",
    "train_valid": "train_valid_aggregated_deterministic.xlsx",
    "test_raw": "test.xlsx",
    "train_valid_raw": "train_valid.xlsx",
}


def load_dataset(name="test", *, directory=None):
    """Read a dataset alias or CSV/XLSX path; return a fresh DataFrame.

    ``train_valid`` is the 291-row reference set. ``test`` contains 38 rows.
    No splitting, fitting, imputation or aggregation is performed by this loader.
    """
    root = Path(directory) if directory is not None else ASSETS
    path = root / DATASETS.get(str(name), str(name))
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")
    return frame
