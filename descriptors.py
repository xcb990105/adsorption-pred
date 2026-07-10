from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FEATURE_NAMES = [
    "Salt anion hydration descriptor",
    "Aanion",
    "Metal ion complex descriptor",
    "Complex binding capacity descriptor",
    "concentration",
    "pH",
]

BLANK_CAPACITY = 142.4037812


class DescriptorError(ValueError):
    pass


def load_descriptor_map(models_dir: Path | str = "models") -> dict:
    path = Path(models_dir) / "descriptor_map.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing descriptor map: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def available_anion_types(descriptor_map: dict) -> list[str]:
    return list(descriptor_map["anion_types"].keys())


def available_ion_types(descriptor_map: dict, anion_type: str) -> list[str]:
    try:
        return list(descriptor_map["anion_types"][anion_type]["ions"].keys())
    except KeyError as exc:
        raise DescriptorError(f"Unknown salt anion type: {anion_type}") from exc


def build_feature_frame(
    descriptor_map: dict,
    anion_type: str,
    ion_type: str,
    concentration: float,
    ph_value: float,
) -> pd.DataFrame:
    try:
        descriptor_values = descriptor_map["anion_types"][anion_type]["ions"][ion_type]["features"]
    except KeyError as exc:
        raise DescriptorError(f"No descriptor mapping for {ion_type} in {anion_type}.") from exc

    row = dict(descriptor_values)
    row["concentration"] = float(concentration)
    row["pH"] = float(ph_value)

    return pd.DataFrame([[row[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)


def relative_change_percent(capacity: float, blank_capacity: float = BLANK_CAPACITY) -> float:
    return (float(capacity) - blank_capacity) / blank_capacity * 100.0
