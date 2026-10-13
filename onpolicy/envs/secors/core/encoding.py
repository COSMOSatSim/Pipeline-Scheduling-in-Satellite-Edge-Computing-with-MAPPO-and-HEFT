"""Traduzione stato di dominio SEC-RSM ↔ tensori MDP (SEC-MPH)."""
import os

import numpy as np

from devices.core.device import GND
from onpolicy.envs.secors import cost
from onpolicy.envs.secors.core.state import State
from onpolicy.envs.secors.core.observation import Observation


class SECORSEncoder:
    """Encoder stato↔tensori per SEC-MPH."""

    def __init__(self, device_list, devices, n_agents, n_devices, agent_offline_windows,
                 lambda1: float = 1.0, lambda2: float = 0.01,
                 beta: float = 0.5, omega: float = 0.01):
        self._device_list = device_list   # ordine fisso SAT-1...n | SAS | GND
        self._devices     = devices        # dict per cost.evaluate (reward)
        self.n_agents     = n_agents
        self.n_devices    = n_devices
        # Finestre offline PER-AGENTE (una lista per agente), normalizzate a tuple (start, end).
        self._agent_windows = [
            [(w["start"], w["end"]) if isinstance(w, dict) else tuple(w) for w in (wins or [])]
            for wins in agent_offline_windows
        ]
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._beta    = beta
        self._omega   = omega
        self._i_gnd = next((i for i, d in enumerate(device_list)
                            if d.device_type == GND), n_devices - 1)

    # ------------------------------------------------------------------ #
    # Offline
    # ------------------------------------------------------------------ #
    def is_offline(self, e: int, t: float) -> bool:
        """True se t cade in una finestra offline dell'agente e ([start, end), vincolo 11)."""
        return any(start <= t < end for start, end in self._agent_windows[e])

    # ------------------------------------------------------------------ #
    # Stato globale / osservazione locale
    # ------------------------------------------------------------------ #
    def get_state(self, last_cs) -> State:
        return State(
            NQ=[len(d.assigned_task_queue) for d in self._device_list],
            CQ=[d.assigned_task_queue[-1].scheduled_execution_end if d.assigned_task_queue else 0.0
                for d in self._device_list],
            CS=list(last_cs),
            F =[self.is_offline(e, cs) for e, cs in enumerate(last_cs)],
        )

    def attese_per_device(self, task) -> np.ndarray:
        """Per ogni device: 1.0 se assegnargli `task` costringerebbe ad attendere la riapertura di
        un collegamento con la terra, 0.0 altrimenti."""
        attesa = np.zeros(self.n_devices, dtype=np.float32)
        for i, target in enumerate(self._device_list):
            for pid in task.predecessors:
                parent = task.dag.task_chunks[pid]
                source = parent.assigned_device
                requested = parent.scheduled_execution_end
                if source is None or requested is None or source is target:
                    continue
                if source.device_type != GND and target.device_type != GND:
                    continue
                duration = source.get_transmission_time(
                    parent.operation, parent.image_size_px, target)
                start = source.get_available_transmission_start(
                    requested, duration, target)
                if start > requested + 1e-7:
                    attesa[i] = 1.0
                    break
        return attesa

    def build_obs(self, last_cs) -> np.ndarray:
        """Osservazioni locali o_t^e = {ND, CQ, cs_e, f_e} (eq. 21), una per agente."""
        state = self.get_state(last_cs)
        return np.array([
            Observation(ND=state.NQ, CQ=state.CQ,
                        sat_cs=last_cs[e], sat_offline=state.F[e]).to_vector()
            for e in range(self.n_agents)
        ], dtype=np.float32)

    def build_share_obs(self, last_cs) -> np.ndarray:
        """Stato globale s_t = {NQ, CQ, CS, F} (eq. 20) replicato per ogni agente."""
        s_t = self.get_state(last_cs).to_vector()
        return np.array([s_t for _ in range(self.n_agents)], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Maschera azioni (vincolo is_assignable)
    # ------------------------------------------------------------------ #

    # Maschera delle azioni: esclude dalle scelte le assegnazioni che il modello
    # ammette ma il cui costo e' fuori scala rispetto all'episodio. Oggi copre le
    # attese sul collegamento con la terra; il nome resta generico perche' altri
    # casi patologici andrebbero esclusi allo stesso modo.
    #
    # Attiva per difetto. La imposta configura_maschera(): dagli iperparametri
    # dell'ambiente in addestramento, dal run_info.json del modello in inferenza,
    # cosi' i modelli anteriori restano valutati come sono stati addestrati.
    # SECORS_ACTION_MASK=0 la spegne e ha la precedenza su tutto.
    _ACTION_MASK = os.environ.get("SECORS_ACTION_MASK", "1") not in ("", "0")

    @classmethod
    def configura_maschera(cls, attiva: bool) -> bool:
        """Imposta la maschera, salvo quando l'ambiente la forza."""
        if os.environ.get("SECORS_ACTION_MASK") is not None:
            return cls._ACTION_MASK       # l'ambiente ha gia' deciso
        cls._ACTION_MASK = bool(attiva)
        return cls._ACTION_MASK

    def build_available_actions(self, pending_tasks) -> np.ndarray:
        """Vincolo dei task ancorati e, opzionalmente, esclusione delle attese."""
        avail = np.ones((self.n_agents, self.n_devices), dtype=np.float32)
        for e, pending in enumerate(pending_tasks):
            if not pending:
                continue
            task = pending[0]
            if not task.is_assignable:
                avail[e, :] = 0.0
                avail[e, self._device_list.index(task.dag.source_device)] = 1.0
                continue
            if not self._ACTION_MASK:
                continue
            avail[e] = 1.0 - self.attese_per_device(task)
            if not avail[e].any():
                avail[e, :] = 1.0
        return avail

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def compute_reward(self, last_cs, costs=None) -> float:
        """rt = ω·(1/n)Σ_e(-cs_e) + (1-ω)·(-[λ1·β·C_T + λ2·(1-β)·C_E])."""
        r_private = sum(-cs for cs in last_cs) / max(self.n_agents, 1)
        C_T, C_E  = cost.evaluate(self._devices) if costs is None else costs
        r_global  = -(self._lambda1 * self._beta * C_T
                      + self._lambda2 * (1 - self._beta) * C_E)
        return self._omega * r_private + (1 - self._omega) * r_global
