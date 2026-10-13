"""Scheduler casuale."""
#######################################################################################################
# IMPORT
#######################################################################################################
import random
from scheduler.core.scheduler import PolicyScheduler

#######################################################################################################
# RANDOM SCHEDULER
#######################################################################################################
class RandomScheduler(PolicyScheduler):
    """Assegna i task generati a device scelti casualmente. apply(devices, device_scheduler=None):
    round-robin sulla costellazione - al passo t ogni SAT decide in parallelo il device random a cui
    assegnare la propria t-esima task."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def select(self, valid_metrics: list, origin=None) -> dict:
        return self._rng.choice(valid_metrics)

    def apply(self, devices: dict, device_scheduler=None) -> None:
        all_devices = [d for group in devices.values() for d in group]
        satellites  = devices.get("SAT", [])
        T = max((len(s.generated_task_queue) for s in satellites), default=0)
        for t in range(T):
            for e in satellites:
                if t < len(e.generated_task_queue):
                    task = e.generated_task_queue[t]
                    # Override per task non-assegnabili (es.
                    if task.is_assignable:
                        device = self._rng.choice(all_devices)
                    else:
                        device = task.dag.source_device
                    device.assign_task(task)

            if device_scheduler is not None:
                device_scheduler.apply(devices)
