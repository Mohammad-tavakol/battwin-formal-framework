"""Saved preprocessing and exact input alignment; no fitting occurs here."""

from pathlib import Path
import warnings
import pandas as pd
from joblib import load
from sklearn.exceptions import InconsistentVersionWarning
from ..schema import ALIASES


def load_saved(path):
    # Record version mismatches once per file instead of once per RF tree.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = load(path)
    versions = sorted(
        {
            (w.message.original_sklearn_version, w.message.current_sklearn_version)
            for w in caught
            if isinstance(w.message, InconsistentVersionWarning)
        }
    )
    if versions:
        warnings.warn(
            f"{path.name}: saved/runtime sklearn versions {versions}", stacklevel=2
        )
    return model


def model_frame(features, columns, preprocessor):
    """Exact training column names; resolve aliases and stage-specific categories."""
    source = (
        features if isinstance(features, pd.DataFrame) else pd.DataFrame([features])
    )
    frame = pd.DataFrame(index=source.index)
    for col in columns:
        canonical = ALIASES.get(col, col)
        if canonical not in source:
            raise KeyError(f"Missing model input: {canonical}")
        frame[col] = source[canonical]
    # The source workbooks use both single/double quoted 'none'. Mixing's encoder
    # learned only the single-quoted spelling. Keep already-known values exact;
    # map an unknown quoted/plain equivalent only to an unambiguous saved category.
    for _, transformer, cols in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        encoder = transformer.named_steps.get("onehot")
        if encoder is None:
            continue
        for col, categories in zip(cols, encoder.categories_):
            known = set(categories)

            def convert(value):
                if pd.isna(value) or value in known:
                    return value
                matches = [
                    c
                    for c in categories
                    if str(c).strip("\"'") == str(value).strip("\"'")
                ]
                return matches[0] if len(matches) == 1 else value

            frame[col] = frame[col].map(convert)
    return frame
