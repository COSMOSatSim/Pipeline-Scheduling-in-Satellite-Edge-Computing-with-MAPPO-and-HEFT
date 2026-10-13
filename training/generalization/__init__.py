"""training/generalization/ - generalizzazione multi-scenario della policy MAPPO."""

import os
import sys

# Metodi di AGGREGAZIONE: partono subito sui modelli per-scenario già addestrati (random_domain non
# è qui: è un training, orchestrato dal launcher).
_AGGREGATION_MODES = ("fedavg", "greedy_soup", "lambda_scaling")


class _Registro:
    """Ripete su un file tutto cio' che il metodo stampa a video."""

    def __init__(self, originale):
        self._originale = originale
        self.righe = []

    def write(self, testo):
        self._originale.write(testo)
        self.righe.append(testo)
        return len(testo)

    def flush(self):
        self._originale.flush()

    def __getattr__(self, nome):          # isatty, encoding e simili
        return getattr(self._originale, nome)


def _cartella_uscita(config: dict, mode: str):
    """Dove il metodo salva il modello, e quindi dove va il registro."""
    from pathlib import Path

    from utility.file_utility import DATA_PATH

    gen = config["GENERALIZATION"]
    return (Path(DATA_PATH) / "models" / "mappo" / gen["models_folder"]
            / gen.get("output_folder", "generalization") / mode)


def _maschera_dei_modelli(config: dict, mode: str) -> bool:
    """La restrizione con cui gli specialisti da aggregare sono stati addestrati."""
    import json
    from collections import Counter
    from pathlib import Path

    from utility.file_utility import DATA_PATH

    gen = config["GENERALIZATION"]
    quanti = int(gen["Settings"][mode]["n_models"])
    radice = Path(DATA_PATH) / "models" / "mappo" / gen["models_folder"]

    dichiarate = {}
    for i in range(quanti):
        info = radice / f"scenario_{i}" / "run_info.json"
        if not info.is_file():
            # Il campo nasce con la maschera: un modello che non lo ha e' stato addestrato prima,
            # quindi senza.
            dichiarate[i] = False
            continue
        with open(info, encoding="utf-8") as f:
            valore = json.load(f).get("POLICY_RESTRICTIONS", {}).get("action_mask")
        dichiarate[i] = bool(valore) if valore is not None else False

    conteggio = Counter(dichiarate.values())
    if len(conteggio) > 1:
        righe = ", ".join(f"scenario_{i}={v}" for i, v in sorted(dichiarate.items()))
        raise RuntimeError(
            "I modelli da aggregare non concordano sulla maschera delle azioni "
            f"({righe}). Aggregarli produrrebbe una politica che non appartiene "
            "ne' all'una ne' all'altra condizione: aggregare separatamente i due "
            "gruppi.")

    return next(iter(conteggio))


def _allinea_maschera(config: dict, mode: str) -> bool:
    """Accende o spegne la maschera secondo quanto i modelli dichiarano."""
    from onpolicy.envs.secors.core.encoding import SECORSEncoder

    attesa = _maschera_dei_modelli(config, mode)
    effettiva = SECORSEncoder.configura_maschera(attesa)
    if effettiva != attesa:
        raise RuntimeError(
            f"I modelli sono stati addestrati con action_mask={attesa}, ma "
            f"SECORS_ACTION_MASK nell'ambiente impone {effettiva}. Rimuovere la "
            "variabile per lasciare decidere ai modelli.")
    print(f"[{mode.upper()}] maschera dedotta dai modelli: "
          f"{'attiva' if attesa else 'spenta'}")
    return attesa


def _intestazione(config: dict, mode: str, maschera: bool) -> str:
    """Da che cosa l'aggregazione e' stata costruita, e con quali parametri."""
    import datetime
    import json

    gen = config["GENERALIZATION"]
    voci = [
        "metodo:             %s" % mode,
        "modelli di origine: %s" % gen["models_folder"],
        "costruita il:       %s" % datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"),
        "maschera:           %s (dedotta dai modelli)"
        % ("attiva" if maschera else "spenta"),
        "parametri:          %s" % json.dumps(gen["Settings"].get(mode, {}),
                                              ensure_ascii=False),
    ]
    riga = "=" * 72
    return riga + "\n" + "\n".join(voci) + "\n" + riga + "\n\n"


def run_method(config: dict, mode: str) -> None:
    """Esegue il metodo di aggregazione `mode` (import lazy del modulo)."""
    if mode == "fedavg":
        from training.generalization.fedavg import run
    elif mode == "greedy_soup":
        from training.generalization.greedy_soup import run
    elif mode == "lambda_scaling":
        from training.generalization.lambda_scaling import run
    else:
        raise ValueError(f"Metodo di aggregazione sconosciuto: '{mode}' "
                         f"(attesi: {', '.join(_AGGREGATION_MODES)}).")

    registro = _Registro(sys.stdout)
    sys.stdout = registro
    try:
        maschera = _allinea_maschera(config, mode)
        run(config)
    finally:
        sys.stdout = registro._originale

    cartella = _cartella_uscita(config, mode)
    cartella.mkdir(parents=True, exist_ok=True)
    percorso = cartella / "log.txt"
    with open(percorso, "w", encoding="utf-8") as f:
        f.write(_intestazione(config, mode, maschera))
        f.write("".join(registro.righe))
    print(f"[{mode.upper()}] registro salvato: {percorso}")
