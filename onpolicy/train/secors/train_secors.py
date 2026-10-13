#!/usr/bin/env python
"""Launcher del training MAPPO."""
import sys
import os
import json
import re

from onpolicy.envs.secors.core.encoding import SECORSEncoder
from onpolicy.train.secors import run_lock
import argparse
from pathlib import Path
import numpy as np
import torch

from onpolicy.utils.load_config import load_config
from onpolicy.envs.core.registry import get_env_class
from onpolicy.envs.core.env_wrappers import make_shared_vec_env
from utility.file_utility import DATA_PATH, CONFIG_PATH, ENV_HP_PATH, load_json
from scenario.scenario import ScenarioFactory


#######################################################################################################
# CONFIG - inietta i parametri del paper (secors.json5) e operativi (training.json5)
#######################################################################################################

def _apply_secors_config(all_args):
    """Inietta i parametri del paper (secors.json5, Table I + RNN) in all_args."""
    cfg = load_json("secors.json5", base_path=ENV_HP_PATH)
    hp  = cfg["PARAMETER SETTING OF SEC-MPH"]
    rnn = cfg["Architettura RNN"]

    all_args.lr                  = hp["lr1"]
    all_args.critic_lr           = hp["lr2"]
    all_args.gamma               = hp["γ"]
    all_args.gae_lambda          = hp["ϱ"]
    all_args.clip_param          = hp["ε"]
    all_args.ppo_epoch           = hp["K"]
    all_args.data_chunk_length   = hp["L"]
    all_args.n_rollout_threads   = hp["B"]
    all_args.seed                = hp["seed"]
    all_args.recurrent_N         = rnn["hp_l"]
    all_args.hidden_size         = rnn["hp_d"]


def _apply_training_settings(all_args, training_cfg):
    """Inietta le impostazioni operative (training.json5) in all_args."""
    all_args.log_interval             = training_cfg.get("print_every", 1)
    all_args.save_interval            = training_cfg.get("checkpoint_every", 1)
    all_args.plot_every               = training_cfg.get("plot_every", 10)
    all_args.print_distribution_every = training_cfg.get("print_distribution_every", 1)


def _apply_from_run_info(all_args, ri):
    """Ripristina gli iperparametri da run_info.json (continuazione di una run).
    """
    m  = ri["MAPPO"]
    ts = ri["TRAINING_SETTINGS"]
    all_args.lr                  = m["lr"]
    all_args.critic_lr           = m["critic_lr"]
    all_args.gamma               = m["gamma"]
    all_args.gae_lambda          = m["gae_lambda"]
    all_args.clip_param          = m["clip_param"]
    all_args.ppo_epoch           = m["ppo_epoch"]
    all_args.data_chunk_length   = m["data_chunk_length"]
    all_args.n_rollout_threads   = m["B_n_rollout"]
    all_args.seed                = m["seed"]
    all_args.entropy_coef        = m["entropy_coef"]
    all_args.use_valuenorm       = m["use_valuenorm"]
    all_args.use_huber_loss      = m["use_huber_loss"]
    all_args.huber_delta         = m["huber_delta"]
    all_args.use_linear_lr_decay = m["use_linear_lr_decay"]
    all_args.recurrent_N         = m["recurrent_N"]
    all_args.hidden_size         = m["hidden_size"]
    all_args.log_interval             = ts.get("print_every", 1)
    all_args.save_interval            = ts.get("checkpoint_every", 1)
    all_args.plot_every               = ts.get("plot_every", 10)
    all_args.print_distribution_every = ts.get("print_distribution_every", 1)


def _finestre(spec) -> str:
    """Finestre offline di un device, in forma compatta."""
    wins = spec.get("offline") or []
    if not wins:
        return "sempre raggiungibile"
    pezzi, da_zero = [], False
    for w in wins:
        a, b = (w["start"], w["end"]) if isinstance(w, dict) else (w[0], w[1])
        pezzi.append("[%g, %g]" % (a, b))
        da_zero = da_zero or a <= 0
    return ("offline dall'avvio " if da_zero else "offline            ") + " ".join(pezzi)


def _carica_stato(state_path):
    """training_state.pt se leggibile, None se assente o troncato."""
    if not state_path.is_file():
        return None
    try:
        return torch.load(state_path, map_location="cpu")
    except Exception:
        return None


def _ultimo_episodio_nel_log(run_dir):
    """Ultimo episodio registrato nel training_log.csv; -1 se il log manca."""
    log = run_dir / "training_log.csv"
    if not log.is_file():
        return -1
    ultima = ""
    with open(log, encoding="utf-8") as f:
        for riga in f:
            if riga.strip():
                ultima = riga
    try:
        return int(ultima.split(",")[0])
    except (ValueError, IndexError):
        return -1


def _stampa_avvio(info, run_dir, EP, da_episodio, maschera, ripreso=False):
    """Riepilogo di che cosa si sta addestrando: scenario, device, parametri."""
    riga = "=" * 78
    sc, mp, rw = info.get("SCENARIO", {}), info.get("MAPPO", {}), info.get("REWARD", {})

    print()
    print(riga)
    print(" ADDESTRAMENTO %s   ·   episodi %d -> %d"
          % ("RIPRESO" if ripreso else "NUOVO", da_episodio + 1, EP))
    print(riga)
    if sc.get("scenario_pool"):
        print(" scenari      pool di %d, campionato a ogni episodio" % len(sc["scenario_pool"]))
        for s in sc["scenario_pool"]:
            print("              %s" % s)
    else:
        print(" scenario     %s" % sc.get("scenario_id", "?"))
    print(" pipeline     %s" % ", ".join(sc.get("pipelines", [])))
    print(" agenti       %-8s task per pipeline  %s"
          % (info.get("num_agents", "?"), info.get("episode_length_T", "?")))
    print(" uscita       %s" % run_dir)

    devices = sc.get("devices")
    if devices:
        print()
        print(" DEVICE")
        for did, spec in devices.items():
            print("   %-7s %-11s %s"
                  % (did, "generatore" if spec.get("generator") else "", _finestre(spec)))

    print()
    print(" RICOMPENSA   λ1=%s  λ2=%s  β=%s  ω=%s"
          % (rw.get("lambda1"), rw.get("lambda2"), rw.get("beta"), rw.get("omega")))
    print(" MAPPO        lr=%s  γ=%s  ϱ=%s  ε=%s  B=%s  seed=%s"
          % (mp.get("lr"), mp.get("gamma"), mp.get("gae_lambda"),
             mp.get("clip_param"), mp.get("B_n_rollout"), mp.get("seed")))
    print("              entropia=%s  huber δ=%s  valuenorm=%s  GRU %s×%s"
          % (mp.get("entropy_coef"), mp.get("huber_delta"), mp.get("use_valuenorm"),
             mp.get("recurrent_N"), mp.get("hidden_size")))
    print(" MASCHERA     %s"
          % ("attiva - le attese fuori scala sono escluse dalle azioni"
             if maschera else "spenta - tutte le assegnazioni sono esplorabili"))
    print(riga)
    print(flush=True)


def _write_run_info(path, run_info):
    """Scrive run_info.json con riga vuota tra le sezioni di primo livello e i
    device dello scenario compatti (un device per riga, come nei file scenario)."""
    devices = run_info.get("SCENARIO", {}).get("devices")
    placeholders = {}
    if devices:
        run_info = json.loads(json.dumps(run_info))   # copia: non tocca il dict del chiamante
        for i, (dev_id, spec) in enumerate(devices.items()):
            marker = f"@@DEVICE_{i}@@"
            placeholders[marker] = json.dumps(spec, ensure_ascii=False)
            run_info["SCENARIO"]["devices"][dev_id] = marker
    text = json.dumps(run_info, indent=2, ensure_ascii=False)
    # riga vuota solo prima delle chiavi di primo livello (non dentro gli oggetti annidati)
    text = re.sub(r'\n  "', '\n\n  "', text)
    for marker, compact in placeholders.items():
        text = text.replace(f'"{marker}"', compact)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_cli(args, all_args):
    """Override da riga di comando, scritti dentro all_args (default None = usa i file di config)."""
    parser = argparse.ArgumentParser(description="train_secors")
    parser.add_argument("--env",              type=str, default=None, help="override environment (default da training.json5)")
    parser.add_argument("--algorithm",        type=str, default=None, help="override algoritmo / cartella (default da training.json5)")
    parser.add_argument("--folder",           type=str, default=None, help="override nome cartella run (default da training.json5)")
    parser.add_argument("--scenario",         type=str, default=None, help="override id scenario (default da training.json5)")
    parser.add_argument("--ep_override",      type=int, default=None, help="override episodi EP (per test brevi)")
    cli = parser.parse_known_args(args)[0]
    for key, value in vars(cli).items():
        setattr(all_args, key, value)
    return all_args


def make_train_env(all_args, env_cls):
    def get_env_fn(rank):
        def init_env():
            env = env_cls(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return make_shared_vec_env([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


#######################################################################################################
# MAIN
#######################################################################################################

def main(args, training_cfg=None):
    # Ricetta canonica MAPPO (mappo.json5) + override CLI.
    all_args = load_config()
    all_args = parse_cli(args, all_args)

    # --- Selettori: env / algoritmo / folder (CLI ha priorità su training_cfg) --- training_cfg può
    # arrivare dal chiamante (blocco TRAINING di config.json5); se assente si legge il file
    # training.json5 (entry-point classico di Secors).
    if training_cfg is None:
        training_cfg = load_json("training.json5", base_path=CONFIG_PATH)
    env_name  = all_args.env       or training_cfg.get("env", "secors")
    algorithm = all_args.algorithm or training_cfg.get("algorithm", "mappo")
    folder    = all_args.folder if all_args.folder is not None else training_cfg.get("folder", "")
    folder    = folder.strip()
    all_args.scenario = all_args.scenario or training_cfg.get("scenario", "base_0_70")
    # Random domain: lista di scenari campionati dall'env a ogni episodio.
    all_args.scenario_pool = training_cfg.get("scenario_pool")

    # Identità della run (etichette per logging/runner).
    all_args.env_name        = env_name
    all_args.algorithm_name  = algorithm
    all_args.experiment_name = env_name

    env_cls = get_env_class(env_name)

    # --- Cartella della run: data/models/<algorithm>/<folder> (folder vuoto -> .../<algorithm>) ---
    run_dir = Path(DATA_PATH) / "models" / algorithm
    if folder:
        run_dir = run_dir / folder
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    # Presidia la cartella: due processi sugli stessi checkpoint li corrompono.
    run_lock.acquisisci(run_dir)

    run_info_path  = run_dir / "run_info.json"
    state_path     = run_dir / "models" / "training_state.pt"
    actor_ckpt     = run_dir / "models" / "actor.pt"

    # --- Auto-detect: continua / completato / nuovo --- CONTINUA -> parametri ed EP dalla run
    # originale (run_info.json): secors.json5 NON si legge.
    run_esistente = run_info_path.is_file() and actor_ckpt.is_file()
    stato         = _carica_stato(state_path) if run_esistente else None
    # Pesi presenti ma stato perduto: la ripresa non puo' ricostruire optimizer e ValueNorm, e
    # trattare la cartella come nuova sovrascriverebbe i checkpoint.
    if run_esistente and stato is None and os.environ.get("SECORS_RIPRESA_DAI_PESI") != "1":
        run_lock.rilascia(run_dir)
        raise RuntimeError(
            "La cartella %s contiene i pesi di una run, ma training_state.pt manca "
            "o non e' leggibile: optimizer e ValueNorm non sono ricostruibili e un "
            "training nuovo sovrascriverebbe actor.pt e critic.pt. Per proseguire "
            "dai soli pesi, con optimizer e ValueNorm azzerati, rilanciare con "
            "SECORS_RIPRESA_DAI_PESI=1." % run_dir)
    has_checkpoint = run_esistente and stato is not None
    ripresa_pesi   = run_esistente and stato is None
    secors_hp      = None   # letto solo per una run nuova

    if has_checkpoint or ripresa_pesi:
        with open(run_info_path, encoding="utf-8") as f:
            saved_info = json.load(f)
        EP = int(saved_info["MAPPO"]["EP_episodi"])      # EP della run, dal suo run_info.json
        if all_args.ep_override is not None:
            EP = all_args.ep_override
        ep_salvato = (int(stato.get("episode", -1)) if has_checkpoint
                      else _ultimo_episodio_nel_log(run_dir))
        if ripresa_pesi and ep_salvato < 0:
            run_lock.rilascia(run_dir)
            raise RuntimeError(
                "Ripresa dai pesi richiesta per %s, ma training_log.csv manca o e' "
                "vuoto: non si sa da quale episodio proseguire." % run_dir)
        if ep_salvato + 1 >= EP:
            print(f"[train_secors] Training COMPLETATO: episodio {ep_salvato + 1}/{EP} in {run_dir}. "
                  f"Per allenare oltre, aumenta EP_episodi nel run_info.json della run.")
            run_lock.rilascia(run_dir)
            return
        # Riprendere una cartella con uno scenario diverso da quello che l'ha generata significa
        # proseguire pesi altrui su un altro problema: i checkpoint verrebbero sovrascritti in
        # silenzio e il modello non corrisponderebbe piu' a niente.
        sc_salvato = saved_info.get("SCENARIO", {}).get("scenario_id")
        if sc_salvato is not None and sc_salvato != all_args.scenario:
            run_lock.rilascia(run_dir)
            raise RuntimeError(
                "La cartella %s contiene una run addestrata su '%s', ma la "
                "configurazione corrente indica '%s'. Riprenderla sovrascriverebbe "
                "quei checkpoint con un addestramento diverso. Correggere "
                "TRAINING.scenario, oppure scegliere un'altra cartella."
                % (run_dir, sc_salvato, all_args.scenario))

        _apply_from_run_info(all_args, saved_info)       # iperparametri della run originale
        # La restrizione appartiene alla run, non alla configurazione corrente: riprenderla diversa
        # produrrebbe una politica addestrata a meta' su due spazi d'azione diversi.
        salvata = saved_info.get("POLICY_RESTRICTIONS", {}).get("action_mask")
        if salvata is not None and SECORSEncoder.configura_maschera(salvata) != bool(salvata):
            raise RuntimeError(
                "La run e' stata avviata con action_mask=%s, ma SECORS_ACTION_MASK "
                "nell'ambiente impone il contrario. Rimuovere la variabile per "
                "riprendere la run come era." % salvata)
        all_args.secors_resume        = has_checkpoint
        all_args.secors_ripresa_pesi  = ripresa_pesi
        all_args.secors_start_episode = ep_salvato + 1
        all_args.model_dir            = str(run_dir / "models")
        _stampa_avvio(saved_info, run_dir, EP, ep_salvato, SECORSEncoder._ACTION_MASK,
                      ripreso=True)
    else:
        secors_env = load_json("secors.json5", base_path=ENV_HP_PATH)
        secors_hp  = secors_env["PARAMETER SETTING OF SEC-MPH"]
        SECORSEncoder.configura_maschera(
            secors_env.get("Maschera azioni", {}).get("enabled", True))
        EP = int(secors_hp["EP"])
        if all_args.ep_override is not None:
            EP = all_args.ep_override
        _apply_secors_config(all_args)
        _apply_training_settings(all_args, training_cfg)
        all_args.secors_resume = False
        all_args.model_dir     = None


    # device
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # probe env per ricavare n agenti e T (lunghezza episodio)
    probe = env_cls(all_args)
    num_agents = probe.n
    n_sat      = getattr(probe, "n_sat", num_agents)
    all_args.episode_length = probe.T
    all_args.num_env_steps  = EP * probe.T * all_args.n_rollout_threads
    probe.close()

    # run_info.json - unica fonte di verità per riproducibilità e continuazione.
    if not (has_checkpoint or ripresa_pesi):
        # Documenta lo scenario per esteso in run_info (riproducibilità).
        scenario = ScenarioFactory.load(all_args.scenario)
        if getattr(all_args, "scenario_pool", None):
            # Random domain: si campiona dal pool, non esiste UNO scenario della run (le pipeline
            # sono comuni a tutto il pool; i device variano per scenario).
            scenario_section = {
                "scenario_pool": list(all_args.scenario_pool),
                "pipelines":     scenario["pipelines"],
            }
        else:
            scenario_section = {
                "scenario_id":   all_args.scenario,
                "devices":       scenario["devices"],
                "pipelines":     scenario["pipelines"],
            }
        run_info = {
            "env":              env_name,
            "algorithm":        algorithm,
            "num_agents":       num_agents,
            "episode_length_T": all_args.episode_length,
            "SCENARIO":         scenario_section,
            # Coefficienti della reward ibrida (eq. 24-26, da secors.json5).
            "REWARD": {"lambda1": secors_hp.get("λ1"), "lambda2": secors_hp.get("λ2"),
                       "beta": secors_hp.get("β"), "omega": secors_hp.get("ω")},
            "MAPPO": {
                "EP_episodi":        EP,
                "B_n_rollout":       all_args.n_rollout_threads,
                "seed":              all_args.seed,
                "lr":                all_args.lr,
                "critic_lr":         all_args.critic_lr,
                "gamma":             all_args.gamma,
                "gae_lambda":        all_args.gae_lambda,
                "clip_param":        all_args.clip_param,
                "ppo_epoch":         all_args.ppo_epoch,
                "data_chunk_length": all_args.data_chunk_length,
                "entropy_coef":      all_args.entropy_coef,
                "use_valuenorm":     all_args.use_valuenorm,
                "use_huber_loss":    all_args.use_huber_loss,
                "huber_delta":       all_args.huber_delta,
                "use_linear_lr_decay": all_args.use_linear_lr_decay,
                "recurrent_N":       all_args.recurrent_N,
                "hidden_size":       all_args.hidden_size,
            },
            "TRAINING_SETTINGS": {
                "print_every":               training_cfg.get("print_every", 1),
                "checkpoint_every":          training_cfg.get("checkpoint_every", 1),
                "plot_every":                training_cfg.get("plot_every", 10),
                "print_distribution_every":  training_cfg.get("print_distribution_every", 1),
            },
            # Restrizioni della politica: fanno parte della sua definizione e vanno riprodotte in
            # inferenza, altrimenti il modello sceglie azioni che in addestramento non ha mai visto.
            "POLICY_RESTRICTIONS": {
                "action_mask": SECORSEncoder._ACTION_MASK,
            },
        }
        _write_run_info(run_info_path, run_info)
        _stampa_avvio(run_info, run_dir, EP, 0, SECORSEncoder._ACTION_MASK)

    envs = make_train_env(all_args, env_cls)

    config = {
        "all_args":   all_args,
        "envs":       envs,
        "eval_envs":  None,
        "num_agents": num_agents,
        "n_sat":      n_sat,
        "device":     device,
        "run_dir":    run_dir,
    }

    from onpolicy.runner.secors.secors_runner import SECORSRunner
    runner = SECORSRunner(config)
    runner.run()

    envs.close()
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()

    # Chiusura ordinata: il presidio va tolto, altrimenti un job successivo lo troverebbe fresco e
    # si fermerebbe credendo la run ancora viva.
    run_lock.rilascia(run_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
