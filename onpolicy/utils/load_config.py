"""Loader di una ricetta canonica di algoritmo in un Namespace."""
from argparse import Namespace

from utility.file_utility import ALGO_HP_PATH, load_json


def load_config(algorithm: str = "mappo") -> Namespace:
    """Carica config/training/algorithm/<algorithm>.json5 in un Namespace (un attributo per flag)."""
    cfg = load_json(f"{algorithm}.json5", base_path=ALGO_HP_PATH)
    return Namespace(**cfg)
