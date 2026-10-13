"""Dove ciascuna modalita' deposita i suoi risultati."""

import os
import time

import json5

from utility.file_utility import DATA_PATH, RESULTS_PATH, CONFIG_PATH, load_json


def cartella_uscita(config: dict) -> str:
    """Cartella in cui la modalita' indicata da CONTROL_PANEL.run scrive."""
    run = config.get("CONTROL_PANEL", {}).get("run")

    if run == "training":
        tr = config["TRAINING"]
        return os.path.join(DATA_PATH, "models", tr.get("algorithm", "mappo"), tr["folder"])

    if run == "generalization":
        gen = config["GENERALIZATION"]
        return os.path.join(DATA_PATH, "models", "mappo", gen["models_folder"],
                            gen.get("output_folder", "generalization"), gen["method"])

    if run == "evaluation":
        # La cartella e' nella suite: con piu' suite si prende la prima, che e' dove il lavoro
        # comincia.
        suite = config["EVALUATION"]["suite"]
        if not isinstance(suite, str):
            suite = suite[0]
        spec = load_json(os.path.join("evaluation", f"{suite}.json5"))
        return os.path.join(RESULTS_PATH, "evaluation", spec["output"])

    if run == "scenarios":
        return os.path.join(CONFIG_PATH, "scenarios",
                            config["SCENARIO_BUILDER"]["output_folder"])

    return RESULTS_PATH


def _sigla(percorso: str) -> str:
    """Riduce un percorso all'osso: via i prefissi che ricorrono ovunque
    (`SB-`, `P-`, `eval_`) e i segmenti ripetuti, che nella coda occupano
    spazio senza distinguere nulla."""
    pezzi = []
    for p in str(percorso).split("/"):
        if not p:
            continue
        for prefisso in ("SB-", "P-", "eval_", "EVAL-"):
            if p.startswith(prefisso) and len(p) > len(prefisso):
                p = p[len(prefisso):]
        p = p.replace("scenario_", "s")
        if not pezzi or p != pezzi[-1]:
            pezzi.append(p)
    return "-".join(pezzi)


def _sigla_metodo(metodo: str) -> str:
    """`random_domain` -> `RD`, `fedavg` -> `FEDAVG`."""
    parole = str(metodo).split("_")
    if len(parole) > 1:
        return "".join(w[:1] for w in parole).upper()
    return metodo.upper()


def nome_run(config: dict) -> str:
    """Etichetta della run per la coda dello scheduler: `SEC-<MODO>-<cosa>`."""
    run = config.get("CONTROL_PANEL", {}).get("run")

    if run == "training":
        return "SEC-TR-" + _sigla(config["TRAINING"]["folder"])

    if run == "generalization":
        gen = config["GENERALIZATION"]
        return "SEC-GEN-%s-%s" % (_sigla(gen["models_folder"]),
                                  _sigla_metodo(gen["method"]))

    if run == "evaluation":
        suite = config["EVALUATION"]["suite"]
        if not isinstance(suite, str):
            suite = suite[0]
        return "SEC-EVAL-" + _sigla(suite)

    if run == "scenarios":
        return "SEC-SCEN-" + _sigla(config["SCENARIO_BUILDER"]["output_folder"])

    return "SEC-SIM-" + _sigla(
        config.get("SIMULATION", {}).get("SCENARIO", {}).get("scenario", "?"))


def file_log(config: dict) -> str:
    """Log dello scheduler, accanto ai risultati e con marca temporale: le run
    riprese dopo il limite di tempo non si sovrascrivono a vicenda."""
    return os.path.join(cartella_uscita(config),
                        "slurm-%s.log" % time.strftime("%Y%m%d-%H%M%S"))


def carica_config() -> dict:
    """La configurazione su cui ragionare."""
    congelata = os.environ.get("SIM_CONFIG")
    if congelata:
        with open(congelata, "r", encoding="utf-8") as f:
            return json5.load(f)
    return load_json("config.json5")


if __name__ == "__main__":
    import sys
    cfg = carica_config()
    if "--log" in sys.argv:
        print(file_log(cfg))
    elif "--nome" in sys.argv:
        print(nome_run(cfg))
    else:
        print(cartella_uscita(cfg))
