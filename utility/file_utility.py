"""Path del progetto + loader JSON5."""
import json5
import os

#######################################################################################################
# PATH CONSTANTS
#######################################################################################################
# radice del progetto (SECMotionModel/): questo file è in utility/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# percorsi cartelle
DATA_PATH      = os.path.join(_ROOT, "data")                 # output: results/
RESULTS_PATH   = os.path.join(_ROOT, "data", "results")
CONFIG_PATH    = os.path.join(_ROOT, "config")               # tutte le configurazioni
MISSION_PATH   = os.path.join(_ROOT, "config", "mission")    # parametri delle missioni
PIPELINES_PATH = os.path.join(_ROOT, "config", "pipelines")  # definizioni delle pipeline
ENV_HP_PATH    = os.path.join(_ROOT, "config", "training", "env")        # iperparametri env (secors.json5)
ALGO_HP_PATH   = os.path.join(_ROOT, "config", "training", "algorithm")  # ricetta algoritmo (mappo.json5)

#######################################################################################################
# JSON5
#######################################################################################################
def load_json(filename, base_path=CONFIG_PATH):
    path = os.path.join(base_path, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json5.load(f)
