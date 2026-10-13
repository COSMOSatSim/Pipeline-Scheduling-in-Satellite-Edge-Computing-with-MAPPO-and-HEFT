"""Calcolo di CT (vincolo 16 del paper SECORS)."""

#######################################################################################################
# CT - TEMPO TOTALE
#######################################################################################################

def compute_CT(devices: dict) -> float:
    """Tempo totale CT (ms) di completamento dello scheduling - vincolo 16."""
    return max(
        (task.scheduled_execution_end or 0.0
         for group in devices.values()
         for device in group
         for task in device.assigned_task_queue),
        default=0.0,
    )
