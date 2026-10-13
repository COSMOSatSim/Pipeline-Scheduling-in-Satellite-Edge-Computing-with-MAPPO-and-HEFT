
"""Gerarchia base degli scheduler e factory."""
#######################################################################################################
# IMPORT
#######################################################################################################
from abc import ABC, abstractmethod

#######################################################################################################
# BASE SCHEDULER
#######################################################################################################
class Scheduler(ABC):
    """Marker base per tutti gli scheduler"""


class PolicyScheduler(Scheduler):
    """Scheduler di policy: assegna i task generati dai satelliti ai device della costellazione.
    apply(devices, device_scheduler=None) -> None [contratto batch] - modifica in-place le
    assigned_task_queue di tutti i device - se device_scheduler != None, lo invoca per riordinare le
    code locali (tipicamente EFT)."""

    @abstractmethod
    def apply(self, devices: dict, device_scheduler: "DeviceScheduler | None" = None) -> None: ...

    def select(self, valid_metrics: list, origin=None) -> "dict | None":
        raise NotImplementedError(f"{type(self).__name__} non supporta la selezione per-task.")


class DeviceScheduler(Scheduler):
    """
    Scheduler locale: riordina le task all'interno del singolo device.

    apply(devices) -> None  (modifica in-place le assigned_task_queue)
    """

    @abstractmethod
    def apply(self, devices: dict) -> None: ...

#######################################################################################################
# SCHEDULER FACTORY
#######################################################################################################
class SchedulerFactory:
    """Factory per la costruzione degli scheduler da configurazione."""

    @staticmethod
    def build_policy(config: dict) -> PolicyScheduler:
        """Costruisce il PolicyScheduler da config["SIMULATION"]["SCHEDULER"]["policy_scheduler"]."""
        cfg      = config.get("SIMULATION", {}).get("SCHEDULER", {}).get("policy_scheduler", {})
        selected = cfg.get("selected", "baseline")
        sub_cfg  = cfg.get(selected, {})

        # import per-caso: si carica solo la policy selezionata (mappo tira dentro torch/onpolicy,
        # inutile per baseline/random)
        match selected:
            case "baseline":
                from scheduler.policy.baselines import BaselineScheduler
                return BaselineScheduler()
            case "random":
                from scheduler.policy.random_scheduler import RandomScheduler
                seed = sub_cfg.get("seed", 42)
                return RandomScheduler(seed=seed)
            case "mappo":
                from pathlib import Path

                from scheduler.policy.mappo_scheduler import MAPPOScheduler
                run_name = sub_cfg.get("model_folder", "")
                if not run_name:
                    raise ValueError("SCHEDULER.policy_scheduler.mappo.model_folder non specificato (es. \"run1\").")
                # Lo scheduler 'mappo' implica la cartella dell'algoritmo: il training salva
                # actor.pt sotto data/models/<algoritmo>/<folder>/models/.
                root = Path(__file__).resolve().parents[2]   # repo root (.../Secors)
                models_root = root / "data" / "models"
                candidates = [
                    models_root / selected / run_name / "models" / "actor.pt",   # layout <algoritmo>/<folder>
                    models_root / run_name / "models" / "actor.pt",              # run_name già completo (es. "mappo/run1")
                    Path(run_name) / "models" / "actor.pt",                      # path diretto a run dir
                    Path(run_name) / "actor.pt",                                 # path diretto a models dir
                ]
                actor_path = next((str(c) for c in candidates if c.is_file()), None)
                if actor_path is None:
                    raise FileNotFoundError(
                        f"actor.pt non trovato per model_folder='{run_name}'. Cercato in: "
                        + ", ".join(str(c) for c in candidates)
                    )
                return MAPPOScheduler(actor_path)
            case _:
                raise ValueError(f"Policy scheduler '{selected}' non riconosciuto.")

    @staticmethod
    def build_device(config: dict) -> "DeviceScheduler | None":
        """Costruisce il DeviceScheduler da config["SIMULATION"]["SCHEDULER"]["device_scheduler"]."""
        from scheduler.device.eft import EFTScheduler

        cfg      = config.get("SIMULATION", {}).get("SCHEDULER", {}).get("device_scheduler", {})
        selected = cfg.get("selected", "")
        if not selected:
            return None
        match selected:
            case "eft":
                diagnostic = (config.get("SIMULATION", {}).get("MONITOR", {})
                              .get("show_device_scheduler_diagnostic", False))
                return EFTScheduler(diagnostic=diagnostic)
            case _:
                raise ValueError(f"Device scheduler '{selected}' non riconosciuto.")
