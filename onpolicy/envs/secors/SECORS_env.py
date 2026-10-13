"""Environment SECORS per onpolicy (pattern SMAC)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from devices.core.device import DeviceUtils, SAT
from scenario.scenario import ScenarioFactory
from scheduler.device.eft import EFTScheduler
from utility.file_utility import ENV_HP_PATH, load_json
from onpolicy.envs.secors import cost
from onpolicy.envs.secors.core.encoding import SECORSEncoder


class SECORSEnv(gym.Env):
    """Environment SEC-RSM per il training MAPPO (SEC-MPH) sopra onpolicy."""

    def __init__(self, args=None):
        self.args = args

        # --- Config: scenario/i (da args) + coeff. reward (secors) ---
        mappo_cfg   = load_json("secors.json5", base_path=ENV_HP_PATH)["PARAMETER SETTING OF SEC-MPH"]
        scenario_id = getattr(args, "scenario", None) or "base_0_70"
        # Pool di scenari (random domain): con più scenari ogni reset() ne pesca uno a caso; senza
        # pool il comportamento resta quello classico.
        pool = list(getattr(args, "scenario_pool", None) or [scenario_id])

        # Coefficienti reward (eq. 24-26) - passati all'encoder (non del trainer)
        lambda1 = mappo_cfg.get("λ1", 1.0)
        lambda2 = mappo_cfg.get("λ2", 0.01)
        beta    = mappo_cfg.get("β",  0.5)
        omega   = mappo_cfg.get("ω",  0.01)

        # --- Un contesto per scenario (device, pipeline, encoder) ---
        self._contexts = [self._build_context(sid, lambda1, lambda2, beta, omega)
                          for sid in pool]
        self._check_pool_compatible(pool)
        self._rng = np.random.RandomState()   # ri-seedato da seed()
        self._activate(0)

        # --- Dimensioni (uguali per tutto il pool, verificate sopra) ---
        self.n_agents  = len(self._generators)
        self.n_devices = len(self._device_list)
        # Numero di SAT: confine fra gli indici SAT e quelli di SAS/GND nella device list (SAT-1...n
        # | SAS | GND).
        self.n_sat     = sum(1 for d in self._device_list if d.device_type == SAT)
        self.T = self._contexts[0]["T"]

        # --- Contratto onpolicy: n agenti + spazi (uno per agente) ---
        self.n = self.n_agents
        obs_dim   = 2 * self.n_devices + 2                  # ND(n+2) + CQ(n+2) + cs_e + f_e
        state_dim = 2 * self.n_devices + 2 * self.n_agents  # NQ(n+2) + CQ(n+2) + CS(agents) + F(agents)
        self.observation_space       = [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,),   dtype=np.float32) for _ in range(self.n)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32) for _ in range(self.n)]
        self.action_space            = [spaces.Discrete(self.n_devices) for _ in range(self.n)]

        # --- Stato MDP ---
        self._pending_tasks  = []
        self._last_cs        = [0.0] * self.n_agents
        self._newly_assigned = []

    #######################################################################################################
    # CONTESTI - uno scenario attivo alla volta (pool = random domain)
    #######################################################################################################

    @staticmethod
    def _build_context(scenario_id, lambda1, lambda2, beta, omega) -> dict:
        """Costruisce lo stato self-contained di UNO scenario: device (con
        generator+offline), code generate, encoder (finestre offline per-agente)."""
        devices, pipeline = ScenarioFactory.build(scenario_id)
        generators = DeviceUtils.generators(devices)
        for gen in generators:
            gen.generate_task_queue(pipeline)
        satellite_pipelines = DeviceUtils.collect_satellite_pipelines(devices)
        device_list = DeviceUtils.device_list(devices)
        encoder = SECORSEncoder(
            device_list, devices, len(generators), len(device_list),
            [g.offline_windows for g in generators], lambda1, lambda2, beta, omega,
        )
        return {
            "scenario_id":         scenario_id,
            "devices":             devices,
            "pipeline":            pipeline,
            "generators":          generators,
            "satellite_pipelines": satellite_pipelines,
            "device_list":         device_list,
            "encoder":             encoder,
            "T": max((len(g.generated_task_queue) for g in generators), default=0),
        }

    def _check_pool_compatible(self, pool: list) -> None:
        """Gli scenari del pool devono avere stessa forma (n agenti, n device, T):
        obs/action space e lunghezza episodio sono fissati alla costruzione."""
        ref = self._contexts[0]
        for sid, ctx in zip(pool[1:], self._contexts[1:]):
            shape     = (len(ctx["generators"]), len(ctx["device_list"]), ctx["T"])
            ref_shape = (len(ref["generators"]), len(ref["device_list"]), ref["T"])
            if shape != ref_shape:
                raise ValueError(
                    f"Scenario '{sid}' incompatibile col pool: (agenti, device, T)="
                    f"{shape} vs {ref_shape} di '{ref['scenario_id']}'.")

    def _activate(self, index: int) -> None:
        """Rende attivo il contesto `index` (scenario corrente dell'episodio)."""
        ctx = self._contexts[index]
        self.devices             = ctx["devices"]
        self.pipeline            = ctx["pipeline"]
        self._generators         = ctx["generators"]
        self.satellite_pipelines = ctx["satellite_pipelines"]
        self._device_list        = ctx["device_list"]
        self.encoder             = ctx["encoder"]

    #######################################################################################################
    # GYM API 
    #######################################################################################################

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self._rng = np.random.RandomState(seed)

    def reset(self):
        # Random domain: l'episodio pesca uno scenario dal pool.
        if len(self._contexts) > 1:
            self._activate(self._rng.randint(len(self._contexts)))
        for group in self.devices.values():
            for device in group:
                device.reset()
        for pipeline in self.satellite_pipelines:
            pipeline.reset()
        self._pending_tasks = [list(p.task_chunks.values()) for p in self.satellite_pipelines]
        self._last_cs       = [0.0] * self.n_agents
        return (self.encoder.build_obs(self._last_cs),
                self.encoder.build_share_obs(self._last_cs),
                self.encoder.build_available_actions(self._pending_tasks))

    def step(self, actions):
        # actions: array (n_agents,) o (n_agents, 1) di indici device
        self._newly_assigned = []
        for e in range(self.n_agents):
            if not self._pending_tasks[e]:
                continue
            task = self._pending_tasks[e].pop(0)
            if task.is_assignable:
                idx    = int(np.asarray(actions[e]).flatten()[0])
                device = self._device_list[idx]
            else:
                device = task.dag.source_device   # vincolo fisico (IS-LAD/SEG)
            device.assign_task(task)
            self._newly_assigned.append((e, task))

        # EFT: sequenzia le code locali (riga 18 Algorithm 1)
        EFTScheduler().apply(self.devices)

        # cs_e <- completion time schedulato del task appena assegnato dal satellite e
        for e, task in self._newly_assigned:
            self._last_cs[e] = task.scheduled_execution_end or 0.0

        C_T, C_E  = cost.evaluate(self.devices)
        reward    = self.encoder.compute_reward(self._last_cs, costs=(C_T, C_E))
        rewards   = np.array([[reward]] * self.n_agents, dtype=np.float32)
        done_flag = all(len(q) == 0 for q in self._pending_tasks)
        dones     = np.array([done_flag] * self.n_agents, dtype=bool)
        infos     = [{"C_T": C_T, "C_E": C_E} for _ in range(self.n_agents)]

        return (self.encoder.build_obs(self._last_cs),
                self.encoder.build_share_obs(self._last_cs),
                rewards, dones, infos,
                self.encoder.build_available_actions(self._pending_tasks))

    def close(self):
        pass
