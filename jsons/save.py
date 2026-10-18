"""Saving utilities for ERP scheduling results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

try:
    from .entities import CalculationState
except ImportError:
    from entities import CalculationState  # type: ignore


def save_schedule(
    state_or_data: Union[CalculationState, dict[str, Any]],
    target: Union[str, Path],
    indent: int = 4,
) -> Path:
    """Save calculation schedule to a JSON file matching output_format.json.

    Parameters
    ----------
    state_or_data : CalculationState or dict
        The schedule state object or formatted dictionary.
    target : str or Path
        Destination file path.
    indent : int
        JSON indentation. Default is 4.

    Returns
    -------
    Path
        Path to the saved file.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(state_or_data, CalculationState):
        out_dict = state_or_data.to_dict()
    elif isinstance(state_or_data, dict):
        out_dict = state_or_data
    else:
        raise TypeError(f"Expected CalculationState or dict, got {type(state_or_data)}")

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=indent)

    return target_path

