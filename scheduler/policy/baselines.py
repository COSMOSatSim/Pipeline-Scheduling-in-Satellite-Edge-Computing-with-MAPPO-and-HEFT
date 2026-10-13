"""Scheduler baseline: ogni satellite elabora le proprie task."""
#######################################################################################################
# IMPORT
#######################################################################################################
from scheduler.core.scheduler import PolicyScheduler

#######################################################################################################
# BASELINE SCHEDULER
#######################################################################################################
class BaselineScheduler(PolicyScheduler):
    """Modalità tradizionale di elaborazione satellitare."""

    def apply(self, devices: dict, device_scheduler=None) -> None:
        for satellite in devices.get("SAT", []):
            for task in satellite.generated_task_queue:
                satellite.assign_task(task)
        if device_scheduler is not None:
            device_scheduler.apply(devices)

    def select(self, valid_metrics: list, origin=None) -> "dict | None":
        """Per-task: il task resta sul server d'origine; None se è fuori maschera."""
        return next((m for m in valid_metrics if m['server'] is origin), None)
