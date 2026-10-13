"""PolicyScheduler basato sul modello MAPPO addestrato (onpolicy)."""
#######################################################################################################
# IMPORT
#######################################################################################################
import json
import os

import numpy as np
import torch
import gymnasium as gym

from scheduler.core.scheduler import PolicyScheduler
from onpolicy.utils.load_config import load_config
from onpolicy.algorithms.mappo.networks.actor import Actor
from onpolicy.envs.secors.core.encoding import SECORSEncoder
from devices.core.device import DeviceUtils
from utility.file_utility import ENV_HP_PATH, load_json


def _build_args():
    """Ricostruisce all_args (ricetta canonica mappo.json5 + RNN dal paper), identico al training."""
    args = load_config()
    cfg  = load_json("secors.json5", base_path=ENV_HP_PATH)
    rnn  = cfg["Architettura RNN"]
    args.recurrent_N = rnn["hp_l"]   # hp_l - dal paper (secors.json5)
    args.hidden_size = rnn["hp_d"]   # hp_d - dal paper (secors.json5)
    return args


#######################################################################################################
# MAPPO SCHEDULER
#######################################################################################################
class MAPPOScheduler(PolicyScheduler):
    """Pre-scheduler one-shot CTDE basato sull'Actor addestrato (onpolicy)."""

    def __init__(self, model_path: str = None, state_dict: dict = None):
        """L'actor arriva da `model_path` (actor.pt su disco, caso inferenza) o
        da `state_dict` in memoria (valutazione astratta nel merging, silenziosa)."""
        if (model_path is None) == (state_dict is None):
            raise ValueError("MAPPOScheduler: specificare model_path OPPURE state_dict.")
        self.model_path  = model_path
        self._state_dict = state_dict
        self._verbose    = model_path is not None
        self.args        = _build_args()
        self.device      = torch.device("cpu")
        self.actor       = None   # costruito in apply() quando n_devices è noto
        if self._verbose:
            print(f"[MAPPOScheduler] actor: {model_path}")
            self._verifica_restrizioni()

    def _verifica_restrizioni(self) -> None:
        """Allinea la maschera a come il modello e' stato addestrato."""
        info = os.path.join(os.path.dirname(os.path.dirname(self.model_path)), "run_info.json")
        if not os.path.isfile(info):
            return
        with open(info, encoding="utf-8") as f:
            atteso = json.load(f).get("POLICY_RESTRICTIONS", {}).get("action_mask")
        if atteso is None:
            # Il campo nasce con la maschera: un modello che non lo ha e' stato addestrato prima,
            # quindi senza.
            atteso = False
        prima = SECORSEncoder._ACTION_MASK
        adesso = SECORSEncoder.configura_maschera(atteso)
        if adesso != bool(atteso):
            raise RuntimeError(
                "Il modello e' stato addestrato con action_mask=%s, ma "
                "SECORS_ACTION_MASK nell'ambiente impone %s. Rimuovere la variabile "
                "per lasciare decidere al modello." % (atteso, adesso))
        if adesso != prima:
            print("[MAPPOScheduler] maschera allineata al modello: %s"
                  % ("attiva" if adesso else "spenta"))

    def _build_actor(self, n_devices: int) -> None:
        obs_dim   = 2 * n_devices + 2
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Discrete(n_devices)
        self.actor = Actor(self.args, obs_space, act_space, device=self.device)
        state_dict = (self._state_dict if self._state_dict is not None
                      else torch.load(self.model_path, map_location=self.device))
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        if self._verbose:
            total_norm = sum(p.norm().item() for p in self.actor.parameters())
            print(f"[MAPPOScheduler] Actor caricato | n_devices={n_devices} | norma parametri={total_norm:.4f}")

    @torch.no_grad()
    def apply(self, devices: dict, device_scheduler=None) -> None:
        """Replica il rollout di SEC-MPH (Algorithm 1, righe 11-21) sui device passati."""
        generators = DeviceUtils.generators(devices)   # agenti = device che generano task
        dev_list   = DeviceUtils.device_list(devices)
        n_agents   = len(generators)
        if n_agents == 0:
            return

        n_devices = len(dev_list)
        if self.actor is None:
            self._build_actor(n_devices)

        T       = max(len(g.generated_task_queue) for g in generators)
        pending = [list(g.generated_task_queue) for g in generators]
        last_cs = [0.0] * n_agents

        # Encoder condiviso col training: garantisce obs/maschere identiche a SECORSEnv.
        encoder = SECORSEncoder(dev_list, devices, n_agents, n_devices,
                                [g.offline_windows for g in generators])

        # Hidden state RNN per agente (batch = n_agents) e maschere (1 = nessun reset).
        rnn_states = np.zeros((n_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        masks      = np.ones((n_agents, 1), dtype=np.float32)

        for _t in range(T):
            # (1) Osservazione + maschera per agente (via encoder, identiche al training)
            obs   = encoder.build_obs(last_cs)
            avail = encoder.build_available_actions(pending)

            # (2) Forward decentralizzato batched -> azione (argmax se non stochastic)
            actions, _, rnn_states = self.actor(
                obs, rnn_states, masks,
                available_actions=avail, deterministic=True,
            )
            actions    = actions.detach().cpu().numpy().flatten().astype(int)
            rnn_states = rnn_states.detach().cpu().numpy()

            # (3) Joint action: assegna il task corrente di ogni satellite
            newly_assigned = []
            for e in range(n_agents):
                if not pending[e]:
                    continue
                task = pending[e].pop(0)
                if task.is_assignable:
                    chosen_device = dev_list[int(actions[e])]
                else:
                    chosen_device = task.dag.source_device   # vincolo fisico (IS-LAD/SEG)
                chosen_device.assign_task(task)
                newly_assigned.append((e, task))

            # (4) Sequencing globale (EFT)
            if device_scheduler is not None:
                device_scheduler.apply(devices)

            # (5) Aggiorna cs_e con il completion time schedulato post-EFT
            for e, task in newly_assigned:
                last_cs[e] = task.scheduled_execution_end or 0.0

