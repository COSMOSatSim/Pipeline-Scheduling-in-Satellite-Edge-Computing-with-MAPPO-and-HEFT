"""Aggregazione FedAvg dei modelli addestrati per scenario."""
from pathlib import Path

import torch

from utility.file_utility import DATA_PATH

_ALGORITHM = "mappo"     # sottocartella di data/models/
_METHOD    = "fedavg"    # nome del metodo = sottocartella dell'output


def fedavg(state_dicts: list) -> dict:
    """Media dei pesi."""
    ref = state_dicts[0]
    avg = {}
    for k in ref:
        vals = [sd[k] for sd in state_dicts]
        if torch.is_floating_point(vals[0]):
            avg[k] = sum(v.float() for v in vals) / len(vals)
        else:
            avg[k] = vals[0].clone()
    return avg


def _check_compatible(state_dicts: list) -> None:
    """Verifica che tutti i modelli abbiano stesse chiavi e stesse forme
    (condizione necessaria per mediare i pesi)."""
    ref = state_dicts[0]
    for i, sd in enumerate(state_dicts[1:], start=1):
        if sd.keys() != ref.keys():
            raise ValueError(f"Modello #{i}: chiavi diverse dal riferimento (architettura diversa).")
        for k in ref:
            if sd[k].shape != ref[k].shape:
                raise ValueError(f"Modello #{i}: forma diversa per '{k}' "
                                 f"({tuple(sd[k].shape)} vs {tuple(ref[k].shape)}).")


def load_specialists(models_folder: str, n: int) -> list:
    """Carica gli N actor per-scenario (scenario_0 ... scenario_{n-1}) e ne verifica la
    compatibilità."""
    root  = Path(DATA_PATH) / "models" / _ALGORITHM
    paths = [root / models_folder / f"scenario_{i}" / "models" / "actor.pt" for i in range(n)]

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("actor.pt mancanti:\n  " + "\n  ".join(missing))

    state_dicts = [torch.load(p, map_location="cpu") for p in paths]
    _check_compatible(state_dicts)
    return state_dicts


def save_aggregated(aggregated: dict, models_folder: str, out_folder: str,
                    method: str, tag: str) -> None:
    """Salva l'actor aggregato in <models_folder>/<out_folder>/<method>/models/ e stampa come
    usarlo."""
    root    = Path(DATA_PATH) / "models" / _ALGORITHM
    rel_out = f"{models_folder}/{out_folder}/{method}"
    out     = root / models_folder / out_folder / method / "models" / "actor.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aggregated, out)
    print(f"[{tag}] actor aggregato salvato: {out}")
    print(f"[{tag}] per usarlo -> SCHEDULER.policy_scheduler.mappo.model_folder: \"{rel_out}\"")


def run(config: dict) -> None:
    """Legge il blocco GENERALIZATION, media gli N actor.pt e salva
    l'aggregato in <models_folder>/<output_folder>/fedavg/models/actor.pt."""
    gen = config["GENERALIZATION"]
    models_folder = gen["models_folder"]                          # es. "SB-4096"
    n             = int(gen["Settings"]["fedavg"]["n_models"])    # scenario_0 ... scenario_{n-1}
    out_folder    = gen.get("output_folder", "generalization")    # sotto models_folder

    print(f"[FEDAVG] media di {n} modelli: {models_folder}/scenario_0 ... scenario_{n - 1}")
    state_dicts = load_specialists(models_folder, n)
    aggregated  = fedavg(state_dicts)
    save_aggregated(aggregated, models_folder, out_folder, _METHOD, "FEDAVG")
